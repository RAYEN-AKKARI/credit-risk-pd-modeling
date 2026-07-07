import numpy as np
from sklearn.linear_model import LinearRegression


class SimpleGLMBoost:
    """
    Custom GLMBoost-inspired model for binary classification.

    The model follows an additive boosting logic by iteratively fitting
    weak linear learners on pseudo-residuals.
    """

    def __init__(self, n_estimators=1000, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        F = np.zeros(X.shape[0])

        for _ in range(self.n_estimators):
            prob = self._sigmoid(F)
            residuals = y - prob

            model = LinearRegression()
            model.fit(X, residuals)

            update = model.predict(X)
            F += self.learning_rate * update

            self.models.append(model)

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        F = np.zeros(X.shape[0])

        for model in self.models:
            F += self.learning_rate * model.predict(X)

        prob = self._sigmoid(F)

        return np.vstack([1 - prob, prob]).T

    def predict(self, X, threshold=0.5):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)


class WeightedGLMBoost:
    """
    Experimental weighted GLMBoost-inspired model.

    This model attempts to update predictions using weighted feature-wise
    weak learners at each boosting iteration.
    """

    def __init__(self, n_estimators=1000, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_models = []
        self.feature_weights = []

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        F = np.zeros(X.shape[0])
        n_features = X.shape[1]

        for _ in range(self.n_estimators):
            prob = self._sigmoid(F)
            residuals = y - prob

            models_m = []
            weights_m = []

            for j in range(n_features):
                X_j = X[:, j].reshape(-1, 1)

                model = LinearRegression()
                model.fit(X_j, residuals)

                prediction = model.predict(X_j)
                corr = np.corrcoef(prediction, residuals)[0, 1]

                if np.isnan(corr):
                    corr = 0.0

                weight = abs(corr)

                models_m.append(model)
                weights_m.append(weight)

            weights_m = np.array(weights_m)

            if weights_m.sum() > 0:
                weights_m = weights_m / weights_m.sum()

            update = np.zeros(X.shape[0])

            for j, model in enumerate(models_m):
                X_j = X[:, j].reshape(-1, 1)
                update += weights_m[j] * model.predict(X_j)

            F += self.learning_rate * update

            self.feature_models.append(models_m)
            self.feature_weights.append(weights_m)

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        F = np.zeros(X.shape[0])

        for models_m, weights_m in zip(self.feature_models, self.feature_weights):
            update = np.zeros(X.shape[0])

            for j, model in enumerate(models_m):
                X_j = X[:, j].reshape(-1, 1)
                update += weights_m[j] * model.predict(X_j)

            F += self.learning_rate * update

        prob = self._sigmoid(F)

        return np.vstack([1 - prob, prob]).T

    def predict(self, X, threshold=0.5):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)
