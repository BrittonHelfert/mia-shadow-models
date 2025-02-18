import numpy as np

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data


NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 45000
ATTACK_TEST_DATASET_SIZE = 2500


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 100, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 12, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 100, "Number of epochs to train attack models.")


def get_data():
    """Prepare CIFAR10 data with specific splitting strategy."""
    (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
    
    # Combine all data
    X_full = np.concatenate([X_train_full, X_test_full])
    y_full = np.concatenate([y_train_full, y_test_full])
    
    # Convert to float and normalize
    X_full = X_full.astype("float32") / 255.0
    y_full = tf.keras.utils.to_categorical(y_full)
    
    # Set aside equal sized train/test sets for target model
    target_size = ATTACK_TEST_DATASET_SIZE * 2  # Equal train and test sizes
    indices = np.arange(len(X_full))
    target_indices = np.random.choice(indices, target_size, replace=False)
    shadow_indices = np.setdiff1d(indices, target_indices)
    
    # Split target data into train/test
    target_train_indices = target_indices[:ATTACK_TEST_DATASET_SIZE]
    target_test_indices = target_indices[ATTACK_TEST_DATASET_SIZE:]
    
    X_target_train = X_full[target_train_indices]
    y_target_train = y_full[target_train_indices]
    X_target_test = X_full[target_test_indices]
    y_target_test = y_full[target_test_indices]
    
    # Remaining data for shadow models
    X_shadow = X_full[shadow_indices]
    y_shadow = y_full[shadow_indices]
    
    return (X_target_train, y_target_train), (X_target_test, y_target_test), (X_shadow, y_shadow)


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

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

    # Get split data
    (X_train, y_train), (X_test, y_test), (X_shadow, y_shadow) = get_data()

    # Train the target model
    print("Training the target model...")
    target_model = target_model_fn()
    target_model.fit(
        X_train, y_train, epochs=FLAGS.target_epochs, validation_data=(X_test, y_test), verbose=True
    )

    # Train the shadow models
    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )

    print("Training the shadow models...")
    X_attack, y_attack = smb.fit_transform(
        X_shadow,
        y_shadow,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True
        ),
    )

    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(
        X_attack, y_attack, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    )

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # Compute the attack accuracy.
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print(attack_accuracy)


if __name__ == "__main__":
    app.run(demo)
