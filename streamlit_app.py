import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer  # Added for handling missing values

# Step 1
st.title("Machine Learning Forecasting App")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
categorical_columns = []
X_encoded = pd.DataFrame()

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Step 3
    selected_column = st.selectbox("Select the column to forecast", data.columns)

    # Step 4: Option to remove missing values
    remove_missing_values = st.checkbox("Remove Missing Values?")
    if remove_missing_values:
        data.dropna(inplace=True)  # Remove rows with missing values

    # Step 5
    minimize_imbalance = st.checkbox("Minimize class imbalance?")
    if minimize_imbalance:
        X = data.drop(columns=[selected_column])
        y = data[selected_column]

        sampler = RandomUnderSampler()
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        data = pd.concat([X_resampled, y_resampled], axis=1)

        st.write("Class imbalance minimized.")
        st.write("Previous class distribution:")
        st.write(y.value_counts())
        st.write("Current class distribution:")
        st.write(y_resampled.value_counts())

    st.write(f"Total number of instances: {len(data)}")
    st.write("Frequency of each feature in the selected column:")
    feature_counts = data[selected_column].value_counts()
    st.write(feature_counts)

    # Step 6a
    validation_type = st.selectbox("Select validation type", ["K-Fold", "Train-Test Split", "Leave-One-Out", "Repeated K-Fold"])

    # Show number of folds input only if K-Fold or Repeated K-Fold is selected
    if validation_type in ["K-Fold", "Repeated K-Fold"]:
        num_folds = st.number_input("Enter number of folds", min_value=2, value=5)

    # Step 6c
    model_name = st.selectbox("Select model", ["Logistic Regression", "Random Forest", "SVM", "K-Nearest Neighbors"])

    # Store accuracies, precisions, recalls, and F1-scores for each test
    metrics = []

    # Step 7: Run button
    if st.button("Run"):
        X = data.drop(columns=[selected_column])
        y = data[selected_column]

        # Label encoding for categorical labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # One-hot encoding for categorical features
        categorical_columns = X.select_dtypes(include=["object"]).columns
        X_encoded = pd.get_dummies(X, columns=categorical_columns)

        if validation_type == "K-Fold":
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        elif validation_type == "Leave-One-Out":
            kf = LeaveOneOut()

        elif validation_type == "Repeated K-Fold":
            kf = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=10, random_state=42)

        else:  # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

            if model_name == "Logistic Regression":
                model = LogisticRegression()
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "SVM":
                model = SVC()
            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
            else:
                st.write("Invalid model selection")

            # Model training
            model.fit(X_train, y_train)

            # Model prediction
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
            recall = recall_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
            f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass

            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")

        if validation_type != "Train-Test Split":
            for i, (train_idx, test_idx) in enumerate(kf.split(data), start=1):
                X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
                y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

                # Create user-selected model instance
                if model_name == "Logistic Regression":
                    model = LogisticRegression()
                elif model_name == "Random Forest":
                    model = RandomForestClassifier()
                elif model_name == "SVM":
                    model = SVC()
                elif model_name == "K-Nearest Neighbors":
                    model = KNeighborsClassifier()
                else:
                    st.write("Invalid model selection")

                # Model training
                model.fit(X_train, y_train)

                # Model prediction
                y_pred = model.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
                recall = recall_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
                f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass

                metrics.append((accuracy, precision, recall, f1))

                if len(metrics) >= 5:  # Stop after 5 tests
                    break

            mean_accuracy = sum(accuracy for accuracy, _, _, _ in metrics) / len(metrics)
            mean_precision = sum(precision for _, precision, _, _ in metrics) / len(metrics)
            mean_recall = sum(recall for _, _, recall, _ in metrics) / len(metrics)
            mean_f1 = sum(f1 for _, _, _, f1 in metrics) / len(metrics)

            st.write(f"Mean Accuracy: {mean_accuracy:.2f}")
            st.write(f"Mean Precision: {mean_precision:.2f}")
            st.write(f"Mean Recall: {mean_recall:.2f}")
            st.write(f"Mean F1 Score: {mean_f1:.2f}")

    # Step 8: Save Model
    if st.button("Save Model"):
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "SVM":
            model = SVC()
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        else:
            st.write("Invalid model selection")

        X = data.drop(columns=[selected_column])
        y = data[selected_column]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        categorical_columns = X.select_dtypes(include=["object"]).columns
        X_encoded = pd.get_dummies(X, columns=categorical_columns)

        model.fit(X_encoded, y_encoded)

        # Save the trained model
        model_filename = f"{model_name.lower()}_model.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)

        st.write(f"{model_name} model saved as {model_filename}")

# Step 9
st.header("Test Pretrained Model")
pretrained_model_file = st.file_uploader("Upload pretrained model", type=["pkl"])

if pretrained_model_file is not None:
    pretrained_model_bytes = pretrained_model_file.read()  # Read the bytes of the uploaded file
    pretrained_model = pickle.loads(pretrained_model_bytes)  # Load the model from the bytes

    st.write("Pretrained model loaded")

    # Step 10: Select your dataset
    uploaded_test_file = st.file_uploader("Choose a CSV file for testing", type="csv")

    if uploaded_test_file is not None:
        test_data = pd.read_csv(uploaded_test_file)

    # Step 11: Select the column to predict
    selected_predict_column = st.selectbox("Select the column to predict", test_data.columns)

    # Reorder columns in the same order as used during training
    X_test = test_data.drop(columns=[selected_predict_column])
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns)  # Or use the same encoder objects

    # Check if the column order matches the trained model's column order
    if list(X_test_encoded.columns) != list(X_encoded.columns):
        st.write("Column order in the test data does not match trained model's column order.")
        st.write("Please make sure the columns are in the same order as used during training.")
    else:
        # Step 12: Use the pretrained model for predictions
        y_pred = pretrained_model.predict(X_test_encoded)

        # Step 13: Display predictions or other relevant information
        st.write("Predictions:")
        st.write(y_pred)
