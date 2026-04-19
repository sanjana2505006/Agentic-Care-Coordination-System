import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os

class ModelTrainer:
    def __init__(self, model_path='src/models/model.pkl'):
        self.model_path = model_path
        self.model = None

    def train(self, df, target_col='No-show'):
        """
        Train a Random Forest model on the provided dataframe.
        """
        if target_col not in df.columns:
            # Try alternative name
            if 'No_show' in df.columns:
                target_col = 'No_show'
            else:
                raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "report": classification_report(y_test, y_pred, output_dict=True),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        self.save_model()
        return metrics

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        return False

    def predict(self, df):
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded.")
        
        # Ensure input DF doesn't have the target column if it's there
        if 'No-show' in df.columns:
            df = df.drop(columns=['No-show'])
        if 'No_show' in df.columns:
            df = df.drop(columns=['No_show'])

        probs = self.model.predict_proba(df)[:, 1]
        return probs
