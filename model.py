import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from dataset import Dataset

# Model class obj for each Dataset class obj
class Model:
    def __init__(self, filename):
        self.dataset = Dataset(filename)
        self.dataset.prepare_dataset_for_training_and_testing()

        self.perform_grid_search()
        self.calculate_all_metrics()

    @staticmethod
    def decimal_to_percentage(num):
        return "{:.0%}".format(num)

    # Perform grid search and store best model as class attribute
    def perform_grid_search(self):
        depths = np.arange(5, 21)
        num_leafs = [1, 5, 10, 20, 50, 100]

        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": depths,
            "min_samples_leaf": num_leafs,
        }

        new_tree_clf = DecisionTreeClassifier()
        grid_search = GridSearchCV(
            new_tree_clf, param_grid, cv=10, scoring="accuracy", return_train_score=True
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
