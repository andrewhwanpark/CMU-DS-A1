from scipy.io import arff
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd


class Dataset:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.load_arff_to_df(self.filename)

    # Replace boolean y-data with 0/1 bits
    @staticmethod
    def fit_transform_y_data(Y_data):
        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(Y_data)
        return y

    # Clean x-data by replacing missing values with median
    @staticmethod
    def clean_x_data_with_median(X_data):
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_data)
        x = imputer.transform(X_data)
        return x

    @staticmethod
    def split_data(X_data, Y_data, test_size=0.15, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, Y_data, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    # Load arff file into pandas df using filename
    def load_arff_to_df(self, filename):
        arffFile = arff.loadarff(filename)
        self.df = pd.DataFrame(arffFile[0])

    # Prepare df for training
    def prepare_dataset(self):
        Y_data = self.df.iloc[:, -1].values
        Y_data = Dataset.fit_transform_y_data(Y_data)

        X_copy = self.df.iloc[:, :-1].copy()
        X_data = Dataset.clean_x_data_with_median(X_copy)

        # Store as class attributes
        self.x = X_data
        self.y = Y_data

    def prepare_train_test_split(self):
        X_train, X_test, y_train, y_test = Dataset.split_data(self.x, self.y)

        # Store as class attributes - these can now be used to train model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def prepare_dataset_for_training_and_testing(self):
        self.prepare_dataset()
        self.prepare_train_test_split()
