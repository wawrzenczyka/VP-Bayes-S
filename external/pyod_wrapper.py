class PyODWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train=None):
        self.model.fit(X_train)
        return self

    def score_samples(self, X):
        return -self.model.decision_function(X)
