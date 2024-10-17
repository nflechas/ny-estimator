import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


class NYEstimatorModel:
    def __init__(self, model_path=None, data_path=None):
        """
        Initialize the MLmodel class by loading the pre-trained model from a pickle file.

        :param model_path: Path to the pickle file containing the trained model.
        :param data_path: Path to the preprocessed data CSV file.
        """
        # Define directories
        self.DIR_REPO = Path.cwd().parent.parent
        self.DIR_DATA_PROCESSED = self.DIR_REPO / "data" / "processed"
        self.DIR_MODELS = self.DIR_REPO / "models"

        # Set paths
        self.model_path = (
            model_path or self.DIR_MODELS / "original_simple_classifier.pkl"
        )
        self.data_path = (
            data_path or self.DIR_DATA_PROCESSED / "preprocessed_listings.csv"
        )

        # Load the model if the path exists
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.clf = pickle.load(f)
            print("Model loaded successfully.")
        else:
            self.clf = None
            print("Model not found. Please train the model first.")

        # Initialize other attributes
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.FEATURE_NAMES = [
            "neighbourhood",
            "room_type",
            "accommodates",
            "bathrooms",
            "bedrooms",
        ]
        self.MAP_ROOM_TYPE = {
            "Shared room": 1,
            "Private room": 2,
            "Entire home/apt": 3,
            "Hotel room": 4,
        }
        self.MAP_NEIGHB = {
            "Bronx": 1,
            "Queens": 2,
            "Staten Island": 3,
            "Brooklyn": 4,
            "Manhattan": 5,
        }
        self.classes = [0, 1, 2, 3]
        self.labels = ["Low", "Mid", "High", "Lux"]
        self.maps = {"0.0": "Low", "1.0": "Mid", "2.0": "High", "3.0": "Lux"}

    def load_data(self):
        """
        Load and preprocess the data from the CSV file.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path, index_col=0)
        self.df.dropna(axis=0, inplace=True)

        # Map categorical features
        self.df["neighbourhood"] = self.df["neighbourhood"].map(self.MAP_NEIGHB)
        self.df["room_type"] = self.df["room_type"].map(self.MAP_ROOM_TYPE)

    def train(self):
        """
        Train the RandomForestClassifier on the preprocessed data and save the model.
        """
        self.load_data()

        X = self.df[self.FEATURE_NAMES]
        y = self.df["category"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=1
        )

        # Initialize and train the classifier
        self.clf = RandomForestClassifier(
            n_estimators=500, random_state=0, class_weight="balanced", n_jobs=4
        )
        self.clf.fit(self.X_train, self.y_train)
        print("Model training completed.")

        # Save the trained model
        self.DIR_MODELS.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.clf, f)
        print(f"Model saved at {self.model_path}")

    def test(self):
        """
        Evaluate the trained model on the test set
        """
        test_metrics = dict()
        if self.clf is None:
            raise Exception(
                "Model is not trained or loaded. Please train the model first."
            )

        if self.X_test is None or self.y_test is None:
            raise Exception("Test data not found. Please train the model first.")

        y_pred = self.clf.predict(self.X_test)
        y_proba = self.clf.predict_proba(self.X_test)

        # Compute and print overall accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        test_metrics["accuracy"] = accuracy

        # Compute and print ROC AUC score
        roc_auc = roc_auc_score(self.y_test, y_proba, multi_class="ovr")
        test_metrics["roc_auc"] = roc_auc

        # Feature importances
        importances = {
            self.FEATURE_NAMES[i]: v
            for i, v in enumerate(self.clf.feature_importances_)
        }
        test_metrics["importances"] = importances

        # Confusion matrix
        c = confusion_matrix(self.y_test, y_pred)
        c_normalized = c / c.sum(axis=1).reshape(len(self.classes), 1)
        test_metrics["c_normalized"] = c_normalized

        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        test_metrics["classification_report"] = report

        return test_metrics

    def predict(self, input_data):
        """
        Predict the price category for given input data and return results as JSON.

        :param input_data: A list of dictionaries containing 'id' and feature values.
            Example:
            [
                {
                    "id": 1,
                    "neighbourhood": "Manhattan",
                    "room_type": "Entire home/apt",
                    "accommodates": 2,
                    "bathrooms": 1.0,
                    "bedrooms": 1
                },
                ...
            ]
        :return: JSON string with list of predictions containing 'id' and 'price_category'.
        """
        if self.clf is None:
            raise Exception(
                "Model is not trained or loaded. Please train the model first."
            )

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Preserve the 'id' column
        ids = input_df["id"].tolist()

        # Drop 'id' from features
        features_df = input_df.drop(columns=["id"])

        # Map categorical features
        features_df["neighbourhood"] = features_df["neighbourhood"].map(self.MAP_NEIGHB)
        features_df["room_type"] = features_df["room_type"].map(self.MAP_ROOM_TYPE)

        # Ensure all required features are present
        missing_features = set(self.FEATURE_NAMES) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")

        # Select and order features
        X_new = features_df[self.FEATURE_NAMES]

        # Predict
        predictions = self.clf.predict(X_new)

        # Map predictions to price categories
        price_categories = [self.maps[str(pred)] for pred in predictions]

        # Prepare JSON output
        results = [
            {"id": id_, "price_category": category}
            for id_, category in zip(ids, price_categories)
        ]
        return results


def main():
    # Initialize the model
    model = NYEstimatorModel()

    # Train the model
    print("Starting training process...")
    model.train()

    # Test the model
    print("Starting testing process...")
    model.test()

    # Example data for prediction
    new_data = [
        {
            "id": 101,
            "neighbourhood": "Manhattan",
            "room_type": "Entire home/apt",
            "accommodates": 2,
            "bathrooms": 1.0,
            "bedrooms": 1,
        },
        {
            "id": 102,
            "neighbourhood": "Brooklyn",
            "room_type": "Private room",
            "accommodates": 1,
            "bathrooms": 1.0,
            "bedrooms": 0,
        },
    ]

    # Make predictions
    print("Making predictions on new data...")
    predictions_json = model.predict(new_data)
    print("Predictions:")
    print(predictions_json)


if __name__ == "__main__":
    main()
