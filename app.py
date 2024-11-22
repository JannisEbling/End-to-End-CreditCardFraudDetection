import sys
import os
import certifi
import streamlit as st
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score

from src.exception.exception import CreditCardException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.main_utils.utils import load_object
from src.utils.ml_utils.model.estimator import CreditCardModel
from src.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# Load environment variables
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

# MongoDB connection
ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Set page config
st.set_page_config(page_title="CreditCard  Analyzer", page_icon="🔒", layout="wide")

# Title
st.title("CreditCard  Analyzer 🔒")
st.markdown("---")

# Sidebar
st.sidebar.title("Actions")
action = st.sidebar.radio("Choose an action:", ["Predict", "Train Model"])

if action == "Train Model":
    st.header("Train Model")
    if st.button("Start Training"):
        try:
            with st.spinner("Training in progress..."):
                train_pipeline = TrainingPipeline()
                train_pipeline.run_pipeline()
            st.success("Training completed successfully!")
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            raise CreditCardException(e, sys)

else:  # Predict
    st.header("Make Predictions")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read and display the uploaded data
            df = pd.read_csv(uploaded_file)
            st.subheader("Preview of uploaded data")
            st.dataframe(df.head())

            if st.button("Make Predictions"):
                with st.spinner("Processing..."):
                    # Load model components
                    try:
                        model = load_object("final_model/model.pkl")
                        preprocessor = load_object("final_model/preprocessor.pkl")
                        
                        # Create CreditCardModel instance
                        network_model = CreditCardModel(preprocessor=preprocessor, model=model)
                        logging.info("Model and preprocessor loaded successfully")

                        # Make predictions
                        y_pred = network_model.predict(df)
                        
                        # Create results dataframe with only predictions
                        results_df = pd.DataFrame({"Prediction": y_pred})
                        
                        # Calculate metrics if class column is available
                        if 'Class' in df.columns:
                            y_true = df['Class']
                            accuracy = accuracy_score(y_true, y_pred)
                            f1 = f1_score(y_true, y_pred)
                            
                            st.subheader("Model Performance Metrics")
                            metrics_df = pd.DataFrame({
                                'Metric': ['Accuracy', 'F1 Score'],
                                'Value': [f"{accuracy:.4f}", f"{f1:.4f}"]
                            })
                            st.table(metrics_df)
                        
                        # Save predictions
                        output_path = "prediction_output/output.csv"
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        results_df.to_csv(output_path, index=False)
                        
                        # Display results
                        st.success("Predictions completed successfully!")
                        st.subheader("Predictions")
                        st.write(results_df)
                        
                        # Download button
                        st.download_button(
                            label="Download predictions as CSV",
                            data=results_df.to_csv(index=False).encode("utf-8"),
                            file_name="network_security_predictions.csv",
                            mime="text/csv",
                        )
                        
                    except Exception as e:
                        logging.error(f"Error during prediction: {str(e)}")
                        st.error(f"An error occurred: {str(e)}")
                        raise CreditCardException(e, sys)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            raise CreditCardException(e, sys)
