import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Model class obj for each Dataset class obj
class Model:
    def __init__(self, model_name, dataset):
        self.model_name = model_name
        self.dataset = dataset

        self.create_appropriate_model()
        self.get_param_grid()

        self.perform_grid_search()
        self.calculate_all_metrics()

    @staticmethod
    def decimal_to_percentage(num):
        return "{:.0%}".format(num)

    def create_appropriate_model(self):
        if self.model_name == "SVC":
            self.model = SVC()
        elif self.model_name == "RF":
            self.model = RandomForestClassifier()
        elif self.model_name == "NB":
            self.model = GaussianNB()
        else:
            # Decision Tree
            self.model = DecisionTreeClassifier()

    def get_param_grid(self):
        if self.model_name == "SVC":
            # Tweaked from: https://www.kaggle.com/viznrvn/optimal-parameters-for-svc-using-gridsearch
            self.param_grid = {
                "C": [1, 10, 100],
                "gamma": [1, 0.1, 0.001],
                "kernel": ["linear"],
            }
        elif self.model_name == "RF":
            # Tweaked from: https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv
            self.param_grid = {
                "n_estimators": [200, 500],
                "max_features": ["auto"],
                "max_depth": [4, 5, 6, 7, 8],
                "criterion": ["gini", "entropy"],
            }
        elif self.model_name == "NB":
            # https://medium.com/analytics-vidhya/how-to-improve-naive-bayes-9fa698e14cba
            self.param_grid = {"var_smoothing": np.logspace(0, -9, num=100)}
        else:
            # Decision Tree
            depths = np.arange(5, 21)
            num_leafs = [1, 5, 10, 20, 50, 100]

            self.param_grid = {
                "criterion": ["gini", "entropy"],
                "max_depth": depths,
                "min_samples_leaf": num_leafs,
            }

    # Perform grid search and store best model as class attribute
    def perform_grid_search(self):
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=3,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1,
        )
        grid_search.fit(self.dataset.X_train, self.dataset.y_train)

        # Store best model as class attribute
        self.best_model = grid_search.best_estimator_

    def calculate_optimized_accuracy(self):
        self.optimized_accuracy = str(
            Model.decimal_to_percentage(
                accuracy_score(
                    self.dataset.y_test, self.best_model.predict(self.dataset.X_test)
                )
            )
        )

    def calculate_optimized_train_accuracy(self):
        self.optimized_train_accuracy = str(
            Model.decimal_to_percentage(
                accuracy_score(
                    self.dataset.y_train, self.best_model.predict(self.dataset.X_train)
                )
            )
        )

    def calculate_optimized_f1_score(self):
        self.optimized_f1_score = str(
            Model.decimal_to_percentage(
                f1_score(
                    self.dataset.y_test, self.best_model.predict(self.dataset.X_test)
                )
            )
        )

    def calculate_optimized_train_f1_score(self):
        self.optimized_train_f1_score = str(
            Model.decimal_to_percentage(
                f1_score(
                    self.dataset.y_train, self.best_model.predict(self.dataset.X_train)
                )
            )
        )

    def calculate_all_metrics(self):
        self.calculate_optimized_accuracy()
        self.calculate_optimized_train_accuracy()
        self.calculate_optimized_f1_score()
        self.calculate_optimized_train_f1_score()
