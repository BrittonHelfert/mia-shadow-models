import numpy as np
import csv  # For saving results
import os

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data

from sklearn.metrics import precision_score, recall_score, accuracy_score


NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
ATTACK_TEST_DATASET_SIZE = 2500


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 100, "Max number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 100, "Max number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 100, "Number of shadow models to train.")


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
    """Attack model that takes target model predictions and predicts membership.
    Following the original paper, this attack model is specific to the class of the input.
    Architecture: Single hidden layer (64 units) with ReLU activation and softmax output.
    """
    model = tf.keras.models.Sequential()

    # Single hidden layer with 64 units and ReLU activation
    model.add(layers.Dense(64, activation="relu", input_shape=(NUM_CLASSES,)))

    # Output layer with softmax activation
    model.add(layers.Dense(2, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def demo(argv):
    del argv  # Unused.

    # Define dataset sizes and callbacks
    dataset_sizes = [2500, 5000, 10000, 15000]
    runs_per_size = 10
    
    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Initialize list to collect results
    results = []

    # Set a base random seed for reproducibility
    base_seed = 42

    # Loop over each dataset size
    for size in dataset_sizes:
        print(f"\n=== Running experiments for target training size: {size} ===")
        
        # Set shadow dataset size equal to target model training size
        shadow_dataset_size = size
        
        for run in range(1, runs_per_size + 1):
            print(f"\n--- Run {run}/{runs_per_size} for size {size} ---")

            # Set seed for reproducibility per run
            seed = base_seed + run
            np.random.seed(seed)
            tf.random.set_seed(seed)

            # Get split data for current run
            (X_train, y_train), (X_test, y_test), (X_shadow, y_shadow) = get_data(size)

            # Train the target model
            print("Training the target model...")
            target_model = target_model_fn()
            target_model.fit(
                X_train, y_train,
                epochs=FLAGS.target_epochs,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )

            # Train the shadow models with matching dataset size
            smb = ShadowModelBundle(
                model_fn=target_model_fn,
                shadow_dataset_size=shadow_dataset_size,
                num_models=FLAGS.num_shadows,
            )

            print("Training the shadow models...")
            X_attack, y_attack = smb.fit_transform(
                X_shadow,
                y_shadow,
                fit_kwargs=dict(
                    epochs=FLAGS.target_epochs,
                    callbacks=[early_stopping],
                    verbose=0
                ),
            )

            # Initialize the attack model bundle (one per class)
            amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

            # Fit the attack models
            print("Training the attack models...")
            amb.fit(
                X_attack, y_attack,
                fit_kwargs=dict(
                    epochs=FLAGS.attack_epochs,
                    callbacks=[early_stopping],
                    verbose=0
                )
            )

            # Prepare attack test data
            data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
            data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

            attack_test_data, real_membership_labels = prepare_attack_data(
                target_model, data_in, data_out
            )

            # Compute the attack predictions
            attack_guesses = amb.predict(attack_test_data)

            # Since real_membership_labels are one-hot encoded, convert them to class labels
            real_labels = np.argmax(real_membership_labels, axis=1)

            # Compute metrics
            precision = precision_score(real_labels, attack_guesses, average='binary')
            recall = recall_score(real_labels, attack_guesses, average='binary')
            accuracy = accuracy_score(real_labels, attack_guesses)

            print(f"Run {run} - Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

            # Append the results
            results.append({
                "dataset_size": size,
                "run": run,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy
            })

    # Save the results to a CSV file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, "membership_inference_attack_results.csv")
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["dataset_size", "run", "precision", "recall", "accuracy"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nExperiment results saved to '{csv_file}'.")


if __name__ == "__main__":
    app.run(demo)
