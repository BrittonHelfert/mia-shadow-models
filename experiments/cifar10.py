import os
import numpy as np
import csv
import logging
import tensorflow as tf
import time

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv  # For saving results

from absl import app
from absl import flags

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data

from sklearn.metrics import precision_score, recall_score, accuracy_score

# Directory setup
def get_base_dir():
    """Get the base directory for data and results.
    
    Returns different paths for server vs local development:
    - Server: /data/USER/mia_project
    - Local: ./  (current directory)
    """
    # More specific check for the server environment
    is_server = os.path.exists('/data') and os.name != 'nt'  # 'nt' means Windows
    
    if is_server:
        # We're on the server
        user = os.getenv('USER')
        return f"/data/{user}/mia_project"
    else:
        # We're on local development - use current directory
        return os.path.abspath(os.path.dirname(__file__)) + "/.."

# Directory setup
BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, "data") 
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set Keras data directory
os.environ['KERAS_HOME'] = DATA_DIR

NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
ATTACK_TEST_DATASET_SIZE = 2500

# Configure GPU and mixed precision
def configure_gpu(use_gpu):
    if not use_gpu:
        print("Disabling GPU...")
        tf.config.set_visible_devices([], 'GPU')
    else:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Use larger batch sizes for GPU training
DEFAULT_BATCH_SIZE = 32
GPU_BATCH_SIZE = 256

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 100, "Max number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 100, "Max number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 100, "Number of shadow models to train.")
flags.DEFINE_bool("test_mode", False, "Run a minimal experiment for testing purposes.")


def get_data(dataset_size):
    """Prepare CIFAR10 data.
    
    Args:
        dataset_size (int): Number of samples for target training and testing.
    
    Returns:
        Tuple of numpy arrays: (X_target_train, y_target_train), (X_target_test, y_target_test), (X_shadow, y_shadow)
    """
    (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
    
    # Combine all data
    X_full = np.concatenate([X_train_full, X_test_full])
    y_full = np.concatenate([y_train_full, y_test_full])
    
    # Convert to float and normalize
    X_full = X_full.astype("float32") / 255.0
    y_full = tf.keras.utils.to_categorical(y_full, NUM_CLASSES)
    
    # Set aside equal sized train/test sets for target model
    target_size = dataset_size * 2  # Equal train and test sizes
    indices = np.arange(len(X_full))
    target_indices = np.random.choice(indices, target_size, replace=False)
    shadow_indices = np.setdiff1d(indices, target_indices)
    
    # Split target data into train/test
    target_train_indices = target_indices[:dataset_size]
    target_test_indices = target_indices[dataset_size:]
    
    X_target_train = X_full[target_train_indices]
    y_target_train = y_full[target_train_indices]
    X_target_test = X_full[target_test_indices]
    y_target_test = y_full[target_test_indices]
    
    # Remaining data for shadow models
    X_shadow = X_full[shadow_indices]
    y_shadow = y_full[shadow_indices]
    
    return (X_target_train, y_target_train), (X_target_test, y_target_test), (X_shadow, y_shadow)


def target_model_fn():
    """The architecture of the target (victim) model. Made to paper specifications."""

    model = tf.keras.models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="tanh",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="tanh", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation="tanh"))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        decay=1e-7
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership."""
    model = tf.keras.models.Sequential([
        # Specify input shape explicitly
        tf.keras.layers.Input(shape=(NUM_CLASSES,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def print_progress(message, verbose=True):
    """Print progress messages in a consistent format."""
    if verbose:
        print(f"\n{message}")


def demo(argv):
    del argv  # Unused.

    # Define experiment parameters; override if in test mode.
    if FLAGS.test_mode:
        print_progress("Test mode enabled: Running minimal experiment.")
        dataset_sizes = [250]          # Use a slightly larger target training size to avoid boundary issues
        runs_per_size = 1              # Just one run per dataset size
        target_epochs = 2              # Only run 2 epochs max
        attack_epochs = 2
        num_shadows = 2              # Use 2 shadow models
        batch_size = DEFAULT_BATCH_SIZE
        attack_test_size = 50        # Smaller test size for attack evaluation
        configure_gpu(False)  # Disable GPU for test mode
    else:
        dataset_sizes = [2500, 5000]  # Just 2 sizes
        runs_per_size = 5            # 5 runs each
        target_epochs = 50           # 50 epochs
        attack_epochs = 50           # 50 epochs
        num_shadows = 20             # 20 shadow models
        batch_size = GPU_BATCH_SIZE
        attack_test_size = ATTACK_TEST_DATASET_SIZE
        configure_gpu(True)  # Enable GPU for full experiment

    # Initialize timing variables after parameters are set
    start_time = time.time()
    completed_runs = 0
    total_runs = len(dataset_sizes) * runs_per_size

    # Load previous results if they exist
    csv_file = os.path.join(RESULTS_DIR, "membership_inference_attack_results.csv")
    existing_results = []
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            existing_results = list(reader)
            
    # Convert existing results to set of (size, run) tuples for easy lookup
    completed_experiments = {(int(r['dataset_size']), int(r['run'])) 
                           for r in existing_results}
    
    results = existing_results

    for size in dataset_sizes:
        for run in range(1, runs_per_size + 1):
            run_start = time.time()
            
            # Skip if this experiment was already done
            if (size, run) in completed_experiments:
                print_progress(f"Skipping completed experiment: size={size}, run={run}")
                continue
                
            print_progress(f"Running experiment: size={size}, run={run}")
            # Set seed for reproducibility per run
            seed = 42 + run
            np.random.seed(seed)
            tf.random.set_seed(seed)

            # Get split data for current run
            (X_train, y_train), (X_test, y_test), (X_shadow, y_shadow) = get_data(size)

            # Train the target model with batch size
            print_progress("Training the target model...")
            target_model = target_model_fn()
            target_model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=target_epochs,
                verbose=0
            )

            # Train shadow models with batch size
            smb = ShadowModelBundle(
                model_fn=target_model_fn,
                shadow_dataset_size=size,
                num_models=num_shadows,
            )

            print_progress("Training the shadow models...")
            X_attack, y_attack = smb.fit_transform(
                X_shadow,
                y_shadow,
                fit_kwargs=dict(
                    batch_size=batch_size,
                    epochs=target_epochs,
                    verbose=0
                ),
            )

            # Initialize the attack model bundle (one per class)
            amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

            # Train attack models with batch size
            print_progress("Training the attack models...")
            amb.fit(
                X_attack, y_attack,
                fit_kwargs=dict(
                    batch_size=batch_size,
                    epochs=attack_epochs,
                    verbose=0
                )
            )

            # Prepare attack test data
            data_in = X_train[:attack_test_size], y_train[:attack_test_size]
            data_out = X_test[:attack_test_size], y_test[:attack_test_size]

            attack_test_data, real_membership_labels = prepare_attack_data(
                target_model, data_in, data_out
            )

            # Compute the attack predictions
            attack_guesses = amb.predict(attack_test_data)

            # Since real_membership_labels are one-hot encoded, convert them to class labels
            real_labels = np.argmax(real_membership_labels, axis=1)

            # Compute overall metrics
            precision = precision_score(real_labels, attack_guesses, average='binary')
            recall = recall_score(real_labels, attack_guesses, average='binary')
            accuracy = accuracy_score(real_labels, attack_guesses)

            # Compute per-class metrics
            # Get the true class for each example (from the attack test data)
            true_classes = np.argmax(attack_test_data[:, NUM_CLASSES:], axis=1)
            
            per_class_metrics = {}
            for class_idx in range(NUM_CLASSES):
                # Get indices for this class
                class_mask = (true_classes == class_idx)
                if not any(class_mask):
                    continue
                    
                # Calculate metrics for this class
                class_precision = precision_score(
                    real_labels[class_mask], 
                    attack_guesses[class_mask], 
                    average='binary'
                )
                class_recall = recall_score(
                    real_labels[class_mask], 
                    attack_guesses[class_mask], 
                    average='binary'
                )
                class_accuracy = accuracy_score(
                    real_labels[class_mask], 
                    attack_guesses[class_mask]
                )
                
                per_class_metrics[class_idx] = {
                    'precision': class_precision,
                    'recall': class_recall,
                    'accuracy': class_accuracy
                }

            print(f"Run {run} - Overall Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
            for class_idx, metrics in per_class_metrics.items():
                print(f"Class {class_idx} - Precision: {metrics['precision']:.4f}, "
                      f"Recall: {metrics['recall']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

            # Append the results with per-class metrics
            result_row = {
                "dataset_size": size,
                "run": run,
                "overall_precision": precision,
                "overall_recall": recall,
                "overall_accuracy": accuracy
            }
            
            # Add per-class metrics to the result row
            for class_idx, metrics in per_class_metrics.items():
                result_row.update({
                    f"class_{class_idx}_precision": metrics['precision'],
                    f"class_{class_idx}_recall": metrics['recall'],
                    f"class_{class_idx}_accuracy": metrics['accuracy']
                })
            
            results.append(result_row)

            # Save results after each run
            with open(csv_file, mode='w', newline='') as file:
                # Get all possible field names (some classes might be missing in some runs)
                fieldnames = ["dataset_size", "run", "overall_precision", "overall_recall", "overall_accuracy"]
                for class_idx in range(NUM_CLASSES):
                    fieldnames.extend([
                        f"class_{class_idx}_precision",
                        f"class_{class_idx}_recall",
                        f"class_{class_idx}_accuracy"
                    ])
                
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)

            completed_runs += 1
            run_time = time.time() - run_start
            remaining_runs = total_runs - completed_runs
            est_remaining_time = run_time * remaining_runs / 60  # minutes
            
            print_progress(f"Completed {completed_runs}/{total_runs} runs. "
                         f"Estimated time remaining: {est_remaining_time:.1f} minutes")

    print(f"\nExperiment results saved to '{csv_file}'.")


if __name__ == "__main__":
    app.run(demo)
