import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path to allow standard imports
_current_dir = Path(__file__).resolve().parent
_src_dir = _current_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from data.preprocessor import load_data, preprocess_data
from models.trainer import ModelTrainer
from agent.coordinator import CareCoordinatorAgent

# Page configuration
st.set_page_config(
    page_title="Agentic Care Coordination System",
    page_icon="🏥",
    layout="wide"
)

def main():
    # Sidebar
    st.sidebar.title("🏥 Care System")
    st.sidebar.info("Upload patient data to predict no-show risks and generate care strategies.")
    
    api_key = st.sidebar.text_input("OpenAI API Key (optional for agent)", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Main content
    st.title("🏥 Clinical Appointment No-Show Prediction & Coordination")
    st.markdown("""
    This system uses **Machine Learning** to predict no-show risks and **Agentic AI** to coordinate care interventions.
    """)

    # File Uploader
    st.header("1. Data Analysis & ML Risk Prediction")
    dataset_file = st.file_uploader("Upload your appointment data (CSV)", type=["csv"])

    if dataset_file is not None:
        try:
            # Load and Preprocess
            raw_data = load_data(dataset_file)
            processed_data = preprocess_data(raw_data)
            
            st.success(f"Dataset loaded: {len(processed_data)} records processed.")
            
            # Model Training / Prediction
            trainer = ModelTrainer()
            
            if st.button("Train/Update Model"):
                with st.spinner("Training model..."):
                    metrics = trainer.train(processed_data)
                    st.write("Model trained with ROC-AUC:", round(metrics['roc_auc'], 3))
                    st.json(metrics['report'])
            
            # Predict
            if trainer.load_model():
                # We need to predict on the current data (excluding target if present)
                features = processed_data.copy()
                target_col = 'No-show' if 'No-show' in features.columns else ('No_show' if 'No_show' in features.columns else None)
                
                if target_col:
                    y_true = features[target_col]
                    features_only = features.drop(columns=[target_col])
                else:
                    y_true = None
                    features_only = features

                risks = trainer.predict(features_only)
                processed_data['RiskScore'] = risks
                processed_data['RiskLevel'] = pd.cut(processed_data['RiskScore'], 
                                                   bins=[0, 0.3, 0.7, 1.0], 
                                                   labels=['Low', 'Medium', 'High'])
                
                st.subheader("Predicted Risk Scores")
                cols = st.columns(3)
                cols[0].metric("Avg Risk Score", round(processed_data['RiskScore'].mean(), 2))
                cols[1].metric("High Risk Patients", len(processed_data[processed_data['RiskLevel'] == 'High']))
                cols[2].metric("Model Status", "Active")

                st.dataframe(processed_data[['RiskScore', 'RiskLevel']].head(10).style.background_gradient(cmap='YlOrRd'))

                # --- Milestone 2: Agentic Coordination ---
                st.divider()
                st.header("2. Agentic Care Coordination")
                
                high_risk_patients = processed_data[processed_data['RiskLevel'] == 'High']
                
                if not high_risk_patients.empty:
                    st.write(f"Found {len(high_risk_patients)} high-risk cases. Select one to generate an intervention plan.")
                    
                    selected_idx = st.selectbox("Select Patient Index", high_risk_patients.index)
                    patient_row = high_risk_patients.loc[selected_idx]
                    
                    if st.button("Generate Coordination Plan"):
                        agent = CareCoordinatorAgent()
                        with st.spinner("Agent is analyzing risk factors and guidelines..."):
                            recommendation = agent.run(patient_row.to_dict(), patient_row['RiskScore'])
                            
                        st.subheader("📋 Intervention Strategy")
                        st.markdown(recommendation)
                else:
                    st.info("No high-risk patients identified for agentic intervention.")

            else:
                st.warning("Please train the model first to enable risk prediction.")

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
