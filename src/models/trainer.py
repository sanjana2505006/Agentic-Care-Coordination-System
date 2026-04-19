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
        self.feature_names = None

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
        self.feature_names = X.columns.tolist()

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
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'model' in data:
                    self.model = data['model']
                    self.feature_names = data.get('feature_names')
                else:
                    self.model = data # Fallback for old models
            return True
        return False

    def predict(self, df):
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded.")
        
        # Drop risk columns if they were previously added
        cols_to_drop = ['RiskScore', 'RiskLevel']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # If we have feature names, ensure the input matches
        if self.feature_names:
            # Check for missing features
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Missing features in input: {missing}. Please re-train the model with the new dataset.")
            
            # Select and reorder columns to match training
            df = df[self.feature_names]

        probs = self.model.predict_proba(df)[:, 1]
        return probs
