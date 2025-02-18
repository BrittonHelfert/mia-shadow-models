"""
Scikit-like estimators for the attack model and shadow models.
"""

import sklearn
import numpy as np

from tqdm import tqdm


class ShadowModelBundle(sklearn.base.BaseEstimator):
    """
    A bundle of shadow models.

    :param model_fn: Function that builds a new shadow model
    :param shadow_dataset_size: Size of the training data for each shadow model
    :param num_models: Number of shadow models
    :param seed: Random seed
    :param ModelSerializer serializer: Serializer for the models. If None,
            the shadow models will be stored in memory. Otherwise, loaded
            and saved when needed.
    """

    MODEL_ID_FMT = "shadow_%d"

    def __init__(
        self, model_fn, shadow_dataset_size, num_models=20, seed=42, serializer=None
    ):
        super().__init__()
        self.model_fn = model_fn
        self.shadow_dataset_size = shadow_dataset_size
        self.num_models = num_models
        self.seed = seed
        self.serializer = serializer
        self._reset_random_state()

    def fit_transform(self, X, y, verbose=False, fit_kwargs=None):
        """Train the shadow models and get a dataset for training the attack.

        :param X: Data coming from the same distribution as the target
                  training data
        :param y: Data labels
        :param bool verbose: Whether to display the progressbar
        :param dict fit_kwargs: Arguments that will be passed to the fit call for
                each shadow model.

        .. note::
            Be careful when holding out some of the passed data for validation
            (e.g., if using Keras, passing `fit_kwargs=dict(validation_split=0.7)`).
            Such data will be marked as "used in training", whereas it was used for
            validation. Doing so may decrease the success of the attack.
        """
        self._fit(X, y, verbose=verbose, fit_kwargs=fit_kwargs)
        return self._transform(verbose=verbose)

    def _reset_random_state(self):
        self._prng = np.random.RandomState(self.seed)

    def _get_model_iterator(self, indices=None, verbose=False):
        if indices is None:
            indices = range(self.num_models)
        if verbose:
            indices = tqdm(indices)
        return indices

    def _get_model(self, model_index):
        if self.serializer is not None:
            model_id = ShadowModelBundle.MODEL_ID_FMT % model_index
            model = self.serializer.load(model_id)
        else:
            model = self.shadow_models_[model_index]
        return model

    def _fit(self, X, y, verbose=False, pseudo=False, fit_kwargs=None):
        """Train the shadow models with overlapping but disjoint in/out sets.
        
        Each shadow model gets:
        - An 'in' set that it trains on (members)
        - An 'out' set it never sees (non-members)
        Sets are disjoint within each model but can overlap between different models.
        """
        self.shadow_in_indices_ = []  # Previously shadow_train_indices_
        self.shadow_out_indices_ = [] # Previously shadow_test_indices_

        if self.serializer is None:
            self.shadow_models_ = []

        fit_kwargs = fit_kwargs or {}
        indices = np.arange(X.shape[0])

        for i in self._get_model_iterator(verbose=verbose):
            # For each shadow model, randomly sample equal sized in/out sets
            # Sets are disjoint within each model but can overlap between models
            total_size = self.shadow_dataset_size * 2  # Size needed for in+out sets
            shadow_indices = self._prng.choice(indices, total_size, replace=False)
            
            # Split into equal sized in/out sets
            in_indices = shadow_indices[:self.shadow_dataset_size]  # Members
            out_indices = shadow_indices[self.shadow_dataset_size:] # Non-members
            
            X_in, y_in = X[in_indices], y[in_indices]
            self.shadow_in_indices_.append(in_indices)
            self.shadow_out_indices_.append(out_indices)

            if pseudo:
                continue

            # Train the shadow model only on the 'in' set
            shadow_model = self.model_fn()
            shadow_model.fit(X_in, y_in, **fit_kwargs)
            if self.serializer is not None:
                self.serializer.save(ShadowModelBundle.MODEL_ID_FMT % i, shadow_model)
            else:
                self.shadow_models_.append(shadow_model)

        self.X_fit_ = X
        self.y_fit_ = y
        self._reset_random_state()
        return self

    def _pseudo_fit(self, X, y, verbose=False, fit_kwargs=None):
        self._fit(X, y, verbose=verbose, fit_kwargs=fit_kwargs, pseudo=True)

    def _transform(self, shadow_indices=None, verbose=False):
        """Produce in/out data for training the attack model.

        :param shadow_indices: Indices of the shadow models to use
                for generating output data.
        :param verbose: Whether to show progress
        """
        shadow_data_array = []
        shadow_label_array = []

        model_index_iter = self._get_model_iterator(
            indices=shadow_indices, verbose=verbose
        )

        for i in model_index_iter:
            shadow_model = self._get_model(i)
            train_indices = self.shadow_in_indices_[i]
            test_indices = self.shadow_out_indices_[i]

            train_data = self.X_fit_[train_indices], self.y_fit_[train_indices]
            test_data = self.X_fit_[test_indices], self.y_fit_[test_indices]
            shadow_data, shadow_labels = prepare_attack_data(
                shadow_model, train_data, test_data
            )

            shadow_data_array.append(shadow_data)
            shadow_label_array.append(shadow_labels)

        X_transformed = np.vstack(shadow_data_array).astype("float32")
        y_transformed = np.vstack(shadow_label_array).astype("float32")
        return X_transformed, y_transformed


def prepare_attack_data(target_model, data_in, data_out):
    """Prepare training data for the attack model.

    :param target_model: Model to attack
    :param data_in: Examples that were used to train the target
    :param data_out: Examples that were not used to train the target
    :returns: (attack_features, attack_labels)
    """
    # Compute predictions on in/out examples.
    in_predictions = target_model.predict(data_in[0])
    out_predictions = target_model.predict(data_out[0])

    # Create features for attack model.
    features_in = np.concatenate([in_predictions, data_in[1]], axis=1)
    features_out = np.concatenate([out_predictions, data_out[1]], axis=1)
    attack_features = np.concatenate([features_in, features_out], axis=0)

    # Create labels for attack model (one-hot encoded)
    attack_labels_in = np.zeros((len(data_in[0]), 2))
    attack_labels_in[:, 1] = 1  # [0,1] for members
    attack_labels_out = np.zeros((len(data_out[0]), 2))
    attack_labels_out[:, 0] = 1  # [1,0] for non-members
    attack_labels = np.concatenate([attack_labels_in, attack_labels_out], axis=0)

    return attack_features, attack_labels


class AttackModelBundle(sklearn.base.BaseEstimator):
    """
    A bundle of attack models, one for each target model class.

    :param model_fn: Function that builds a new shadow model
    :param num_classes: Number of classes
    :param ModelSerializer serializer: Serializer for the models. If not None,
            the models will not be stored in memory, but rather loaded
            and saved when needed.
    :param class_one_hot_encoded: Whether the shadow data uses one-hot encoded
            class labels.
    """

    MODEL_ID_FMT = "attack_model_%d"

    def __init__(
        self, model_fn, num_classes, serializer=None, class_one_hot_coded=True
    ):
        self.model_fn = model_fn
        self.num_classes = num_classes
        self.serializer = serializer
        self.class_one_hot_coded = class_one_hot_coded

    def fit(self, X, y, verbose=False, fit_kwargs=None):
        """Train the attack models.

        :param X: Shadow predictions from ShadowBundle.fit_transform
        :param y: One-hot encoded labels indicating membership
        :param verbose: Whether to display the progressbar
        :param fit_kwargs: Arguments passed to model.fit
        """
        X_total = X[:, : self.num_classes]
        classes = X[:, self.num_classes :]

        datasets_by_class = []
        data_indices = np.arange(X_total.shape[0])
        for i in range(self.num_classes):
            if self.class_one_hot_coded:
                class_indices = data_indices[np.argmax(classes, axis=1) == i]
            else:
                class_indices = data_indices[np.squeeze(classes) == i]

            datasets_by_class.append((X_total[class_indices], y[class_indices]))

        if self.serializer is None:
            self.attack_models_ = []

        dataset_iter = datasets_by_class
        if verbose:
            dataset_iter = tqdm(dataset_iter)
        for i, (X_train, y_train) in enumerate(dataset_iter):
            model = self.model_fn()
            fit_kwargs = fit_kwargs or {}
            model.fit(X_train, y_train, **fit_kwargs)

            if self.serializer is not None:
                model_id = AttackModelBundle.MODEL_ID_FMT % i
                self.serializer.save(model_id, model)
            else:
                self.attack_models_.append(model)

    def _get_model(self, model_index):
        if self.serializer is not None:
            model_id = AttackModelBundle.MODEL_ID_FMT % model_index
            model = self.serializer.load(model_id)
        else:
            model = self.attack_models_[model_index]
        return model

    def predict(self, X):
        """Predict membership (0/1) for given examples."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)  # Return class with highest probability

    def predict_proba(self, X):
        """Predict membership probabilities for given examples."""
        result = np.zeros((X.shape[0], 2))
        shadow_preds = X[:, :self.num_classes]
        classes = X[:, self.num_classes:]

        data_indices = np.arange(shadow_preds.shape[0])
        for i in range(self.num_classes):
            model = self._get_model(i)
            if self.class_one_hot_coded:
                class_indices = data_indices[np.argmax(classes, axis=1) == i]
            else:
                class_indices = data_indices[np.squeeze(classes) == i]

            membership_preds = model.predict(shadow_preds[class_indices])
            for j, example_index in enumerate(class_indices):
                result[example_index] = membership_preds[j]

        return result
