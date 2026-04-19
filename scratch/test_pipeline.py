import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_dir))

from data.preprocessor import load_data, preprocess_data
from models.trainer import ModelTrainer

def test_pipeline():
    print("Loading data...")
    df = load_data("data/raw/sample_appointments.csv")
    
    print("Preprocessing...")
    processed_df = preprocess_data(df)
    print(f"Processed shape: {processed_df.shape}")
    
    print("Training model...")
    trainer = ModelTrainer()
    metrics = trainer.train(processed_df)
    print(f"Training complete. ROC-AUC: {metrics['roc_auc']}")
    
    print("Saving model... Done.")

if __name__ == "__main__":
    test_pipeline()
