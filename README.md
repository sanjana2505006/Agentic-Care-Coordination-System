# Clinical Appointment No-Show Prediction and Agentic Care Coordination System

An AI-powered healthcare operations system that predicts patient appointment no-shows using machine learning and extends into an intelligent agent-based assistant for generating actionable care coordination strategies.

## Project Overview

Patient no-shows are a significant issue in healthcare, leading to wasted resources and longer wait times for other patients. Manual risk assessment is inefficient and often inconsistent.

This system automates risk assessment using traditional Machine Learning models and provides a scalable architecture for intelligent health guidance.

The project is divided into two milestones:

**Milestone 1 – ML-Based Appointment No-Show Prediction**
Predicts the likelihood of a patient missing an appointment using historical scheduling data.

**Milestone 2 – Agentic AI Care Coordination Assistant**
Extends the system using an agentic workflow to generate structured intervention recommendations based on predicted risks.

---

## Problem Statement

Healthcare institutions require efficient systems to:
-   Identify high-risk patients early.
-   Maximize resource utilization by reducing no-show rates.
-   Assist care coordinators with data-driven intervention strategies.
-   Maintain consistency in patient communication and follow-up.

This project builds a structured ML-based risk prediction system and extends it into an intelligent decision-support assistant.

---

## Key Features

### Milestone 1 – Machine Learning Risk Prediction

-   **Data Cleaning & Preprocessing**: Handling dates, categorical features, and calculating `LeadTime`.
-   **Feature Engineering**: Creating operational features like "days until appointment".
-   **Multiple ML Models**:
    -   Logistic Regression
    -   Decision Tree / Random Forest
    -   Gradient Boosting (XGBoost/LightGBM)
-   **Model Comparison & Evaluation**: Using Precision, Recall, F1-Score, and ROC-AUC.
-   **Risk Score Generation**: 0-1 probability estimation.
-   **Interactive Streamlit Interface**:
    -   Upload patient data (CSV).
    -   View risk predictions and feature importance.
-   **Public Deployment**: Hosted on Streamlit Community Cloud or Hugging Face Spaces.

### Milestone 2 – Agentic AI Extension

-   **Risk Analysis**: Agents analyze high-risk cases to determine contributing factors.
-   **Knowledge Retrieval**: Fetches best-practice care coordination guidelines (RAG).
-   **Actionable Recommendations**: Generates specific intervention steps (e.g., "Send SMS reminder 2 days before" or "Call to arrange transportation").
-   **Structured Reporting**: Outputs a clear plan for care coordinators.

---

## System Architecture

### Milestone 1 Workflow
1.  **User Input**: Upload appointment data.
2.  **Data Preprocessing**: Clean dates, encode features.
3.  **Feature Engineering**: Calculate `LeadTime`, `PreviousNoShows`.
4.  **ML Model**: Predict probability.
5.  **Risk Score**: Classify as Low/Medium/High Risk.
6.  **UI Display**: Show dashboard and risk table.

### Milestone 2 Workflow
1.  **Risk Prediction Output**: High-risk patients identified.
2.  **Agent Workflow**: LangGraph agent receives patient context.
3.  **Knowledge Retrieval**: Consults guideline database.
4.  **Reasoning**: Determines best intervention strategy.
5.  **Recommendation Generation**: Produces tailored advice.
6.  **UI Display**: Presents recommendations to the care coordinator.

---

## Project Structure

```
├── data/
│   ├── raw/             # Original dataset
│   ├── processed/       # Cleaned data
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── src/
│   ├── app/
│   │   └── main.py      # Streamlit application entry point
│   ├── data/
│   │   └── preprocessor.py # Data loading and cleaning logic
│   ├── features/        # Feature engineering scripts
│   ├── models/          # Model training scripts
│   ├── agent/           # Agentic AI logic
│   └── utils/           # Helper functions
├── tests/               # Unit tests
└── README.md            # Project documentation
```

---

## Installation & Setup

Follow these steps to set up the project on your local machine.

### Prerequisites

-   **Python 3.9+**: Ensure you have Python installed.
-   **pip**: Package installer for Python.

### Step 1: Clone the Repository

```bash
git clone https://github.com/sanjana2505006/Agentic-Care-Coordination-System.git
cd Agentic-Care-Coordination-System
```

### Step 2: Create a Virtual Environment

It is recommended to create a virtual environment to avoid dependency conflicts.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run src/app/main.py
```

## 🚀 How to Run the Project (Beginner's Guide)

Follow these steps to get the system running on your local machine.

### 1. Prerequisites
- **Python 3.9 or higher** installed on your system.
- An **OpenAI API Key** (optional, but recommended for the Agentic AI features).

### 2. Setup the Environment
Open your terminal and run the following:

```bash
# Clone the repository
git clone https://github.com/sanjana2505006/Agentic-Care-Coordination-System.git
cd Agentic-Care-Coordination-System

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Launch the Streamlit dashboard:
```bash
streamlit run src/app/main.py
```

The application will typically start on `http://localhost:8501`. If port 8501 is busy, it will automatically use the next available port (8502, 8503, etc.).

### 5. Using the System
1. **Upload Data**: Once the app opens, upload the sample file located at `data/raw/sample_appointments.csv`.
2. **Train Model**: Click the **"Train/Update Model"** button to initialize the ML prediction engine.
3. **Predict Risks**: The app will automatically calculate risk scores for all patients.
4. **Agentic Coordination**: 
   - Scroll down to the **"Agentic Care Coordination"** section.
   - (Optional) Enter your OpenAI API Key in the sidebar for better reasoning.
   - Select a high-risk patient and click **"Generate Coordination Plan"** to see the AI agent's recommendations.

---

## Troubleshooting

### Common Issues

**1. Port Already in Use**
If you see "Port 8501 is not available", the app will automatically try the next available port. Check the terminal output for the actual URL.

**2. LangChain Import Errors**
If you encounter `ModuleNotFoundError: No module named 'langchain.docstore'`, ensure you have the latest version of langchain packages:
```bash
pip install --upgrade langchain langchain-core langchain-community
```

**3. Missing Dependencies**
If you encounter import errors, reinstall all dependencies:
```bash
pip install -r requirements.txt --upgrade
```

**4. Model Training Issues**
- Ensure your CSV file has the required columns: `ScheduledDay`, `AppointmentDay`, `No-show`
- The system automatically handles both `No-show` and `No_show` column names
- If the model fails to train, check that your data has both shows and no-shows

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: Minimum 4GB RAM recommended
- **Storage**: 500MB free space for dependencies and model files

---
