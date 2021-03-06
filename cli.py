import glob
import os
from PyInquirer import prompt
from tabulate import tabulate
from dataset import Dataset

from model import Model


class CLI:
    def __init__(self):
        self.parse_datasets_folder()
        self.create_cli_questions()
        self.filename_to_smell()

        self.answers = prompt(self.questions)

        # Results to store accuracy and F1 scores
        self.results = []

        self.parse_input()
        self.prepare_table()
        self.print_results()

    def parse_datasets_folder(self):
        # Name of datasets including file extension
        datasets = [os.path.basename(x) for x in glob.glob("datasets/*.arff")]
        self.datasets = datasets

    def create_cli_questions(self):
        choices = []
        for dataset in self.datasets:
            choices.append({"name": dataset})

        questions = [
            {
                "type": "list",
                "name": "action",
                "message": "Select action",
                "choices": ["run", "compare"],
                "validate": lambda answer: "You must choose at least one action."
                if len(answer) == 0
                else True,
            },
            {
                "type": "checkbox",
                "name": "models",
                "message": "Select models",
                "choices": [
                    {"name": "DT"},
                    {"name": "RF"},
                    {"name": "NB"},
                    {"name": "SVC"},
                ],
                "validate": lambda answer: "You must choose at least one model."
                if len(answer) == 0
                else True,
            },
            {
                "type": "checkbox",
                "name": "datasets",
                "message": "Select datasets",
                "choices": choices,
                "validate": lambda answer: "You must choose at least one dataset."
                if len(answer) == 0
                else True,
            },
        ]

        self.questions = questions

    # Returns smell name from filename with extension
    # e.g. feature-envy.arff -> Feature Envy
    def filename_to_smell(self):
        smells = {}
        for filename in self.datasets:
            filename_without_extension = os.path.splitext(filename)[0]
            words = [x.capitalize() for x in filename_without_extension.split("-")]
            smell = " ".join(words)
            smells[filename] = smell

        self.smells = smells

    def parse_input(self):
        action = self.answers["action"]
        datasets_input = self.answers["datasets"]
        models = self.answers["models"]

        for dataset_input in datasets_input:
            dataset_path = os.path.join(os.getcwd(), "datasets", dataset_input)

            # Prepare dataset
            dataset = Dataset(dataset_path)
            dataset.prepare_dataset_for_training_and_testing()

            # Train models
            for model_name in models:
                print(f":::TRAINING {model_name} MODEL:::")
                model = Model(model_name, dataset)
                model_results = {}

                if action == "run":
                    model_results = {
                        "model": model_name,
                        "smell": self.smells.get(dataset_input),
                        "accuracy": model.optimized_accuracy,
                        "f1_score": model.optimized_f1_score,
                    }
                else:
                    # compare
                    model_results = {
                        "model": model_name,
                        "smell": self.smells.get(dataset_input),
                        "accuracy": model.optimized_accuracy,
                        "train_accuracy": model.optimized_train_accuracy,
                        "f1_score": model.optimized_f1_score,
                        "train_f1_score": model.optimized_train_f1_score,
                    }

                self.results.append(model_results)

    def prepare_table(self):
        self.table = tabulate(self.results, headers="keys", tablefmt="pretty")

    def print_results(self):
        print(self.table)
