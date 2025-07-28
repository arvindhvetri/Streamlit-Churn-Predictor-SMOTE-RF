import pandas as pd
import joblib
import os
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# --- Configuration and File Paths ---
REFERENCE_DATASET_PATH = "Churned.csv"
MODEL_PATH = "model.sav"

try:
    # Assuming Reference_Dataset.csv now only contains the features you provided.
    # The 'Churn' column is *not* expected in this specific reference dataset.
    df_1 = pd.read_csv(REFERENCE_DATASET_PATH)

    # If your CSV has an unnamed index column from saving, drop it:
    if df_1.columns[0] == 'Unnamed: 0':
        df_1 = df_1.drop(columns=df_1.columns[0], axis=1)

    # Ensure 'TotalCharges' is numeric for the loaded DataFrame
    df_1['TotalCharges'] = pd.to_numeric(df_1['TotalCharges'], errors='coerce').fillna(0)
    model = joblib.load(open(MODEL_PATH, "rb"))
    #st.success("Reference dataset and model loaded successfully.")

    # Prepare df_1_features: This will be used to get all possible column names
    # after one-hot encoding, matching the model's training features.
    # Since your provided reference data doesn't have a 'Churn' column,
    # we don't need to drop it here.
    df_1_features = df_1.copy()

except FileNotFoundError as e:
    st.error(f"Error: Required file not found. Please ensure '{REFERENCE_DATASET_PATH}' and '{MODEL_PATH}' exist in the same directory as the script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

# --- Derive MODEL_FEATURES for consistent preprocessing ---
# This block must perfectly mirror the preprocessing done during model training
try:
    # Use df_1_features which now contains only the input features
    temp_df_for_features = df_1_features.copy()

    # Apply the same tenure grouping
    # Ensure tenure is an integer for pd.cut
    temp_df_for_features['tenure'] = temp_df_for_features['tenure'].astype(int)
    labels_for_features = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    temp_df_for_features['tenure_group'] = pd.cut(temp_df_for_features.tenure, range(1, 80, 12), right=False, labels=labels_for_features)
    # Drop original tenure as it's replaced by tenure_group
    temp_df_for_features.drop(columns=['tenure'], axis=1, inplace=True)

    # Get one-hot encoded columns from the reference dataset's features
    MODEL_FEATURES = pd.get_dummies(temp_df_for_features).columns.tolist()
    #st.info("Model features derived for consistent prediction.")
except Exception as e:
    st.error(f"Error deriving model features from reference dataset: {e}. Ensure 'tenure' column exists and is numeric.")
    MODEL_FEATURES = [] # Fallback to empty list if error occurs


# --- Streamlit App Layout and Styling ---
st.set_page_config(page_title="Customer Churn Prediction", layout="wide") # Changed to "wide"

st.title("Customer Churn Prediction")

# Create two columns for the main layout
input_column, result_column = st.columns([0.6, 0.4]) # Adjust ratios as needed

with input_column:
    # Create form for user input
    with st.form("churn_prediction_form"):
        st.header("Customer Information")

        # Group inputs into columns for better layout
        col1, col2, col3 = st.columns(3)

        with col1:
            senior_citizen = st.selectbox("Senior Citizen", ["0", "1"], help="Is the customer a senior citizen? (0 for No, 1 for Yes)")
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0, step=0.01, help="The amount charged to the customer monthly.")
            total_charges = st.number_input("Total Charges", min_value=0.0, value=100.0, step=0.01, help="The total amount charged to the customer.")
            gender = st.selectbox("Gender", ["Female", "Male"])
            partner = st.selectbox("Partner", ["Yes", "No"])

        with col2:
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

        with col3:
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        # Remaining inputs outside columns for full width
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1, step=1, help="Number of months the customer has stayed with the company.")

        submit_button = st.form_submit_button("Predict Churn")

with result_column:
    if submit_button:
        if not MODEL_FEATURES:
            st.error("Model features could not be loaded. Prediction cannot proceed. Please check the reference dataset and ensure 'tenure' column is present.")
        else:
            try:
                # Create a DataFrame for the new input
                new_data_dict = {
                    'SeniorCitizen': [int(senior_citizen)],
                    'MonthlyCharges': [float(monthly_charges)],
                    'TotalCharges': [float(total_charges)],
                    'gender': [gender],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'tenure': [int(tenure)]
                }
                new_df = pd.DataFrame(new_data_dict)

                # --- Consistent preprocessing for new_df (single prediction) ---
                # This must perfectly mirror the preprocessing done during model training and MODEL_FEATURES generation

                # 1. Handle tenure_group for the new_df
                labels_predict = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
                new_df['tenure_group'] = pd.cut(new_df.tenure.astype(int), range(1, 80, 12), right=False, labels=labels_predict)
                new_df.drop(columns=['tenure'], axis=1, inplace=True) # Drop original tenure after creating tenure_group

                # 2. Concatenate with df_1_features to ensure all possible categories are present for one-hot encoding
                # This is crucial for get_dummies to produce all expected columns, even if a category isn't in new_df
                # df_1_features already has tenure_group applied, so we need to ensure new_df is similarly prepared
                # for proper concatenation of categorical features before get_dummies
                combined_for_dummies_temp = pd.concat([df_1_features.drop(columns=['tenure']).assign(tenure_group=pd.cut(df_1_features['tenure'], range(1, 80, 12), right=False, labels=labels_predict)), new_df], ignore_index=True)


                # 3. One-hot encode all relevant columns
                processed_combined_df = pd.get_dummies(combined_for_dummies_temp)

                # 4. Select only the last row (the new input) and align its columns with MODEL_FEATURES
                single_input_processed = processed_combined_df.tail(1).reindex(columns=MODEL_FEATURES, fill_value=0)

                # Make prediction
                single_prediction = model.predict(single_input_processed)[0]
                probability = model.predict_proba(single_input_processed)[:, 1][0] # Probability of churn (class 1)

                st.subheader("Prediction Result")
                if single_prediction == 1:
                    st.error(f"**Prediction:** This customer is likely to churn!")
                    st.info(f"**Confidence:** {probability * 100:.2f}%")
                else:
                    st.success(f"**Prediction:** This customer is likely to continue.")
                    st.info(f"**Confidence:** {(1 - probability) * 100:.2f}%") # Probability of not churn (class 0)

            except ValueError as ve:
                st.error(f"Input Error: Please ensure all numerical fields are valid numbers and all fields are filled. Details: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
                st.exception(e) # Display full traceback for debugging