import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# App config
st.set_page_config(page_title="Insurance Charges Prediction", layout="wide")
st.title("🔮 Insurance Charges Prediction")

# File to store predictions
PREDICTIONS_FILE = "saved_predictions.csv"

# Create predictions file if it doesn't exist
if not os.path.exists(PREDICTIONS_FILE):
    pd.DataFrame(columns=["Age", "Sex", "BMI", "Children", "Smoker", "Region", "Predicted Charges"]).to_csv(PREDICTIONS_FILE, index=False)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

# Train model
@st.cache_resource
def train_model(data):
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])

    X = data[["age", "sex", "bmi", "children", "smoker", "region"]]
    y = data["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return pipeline, r2, X_test, y_test

# Load and train
data = load_data()
model, model_accuracy, X_test, y_test = train_model(data)

# Sidebar inputs
st.sidebar.header("📝 Enter Your Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input("Children", min_value=0, max_value=10, value=0)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", sorted(data["region"].unique()))

# Prediction logic
if st.sidebar.button("🚀 Predict Charges"):
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region
    }])

    try:
        prediction_raw = model.predict(input_data)[0]
        prediction = int(round(prediction_raw * 80))  # Convert to Rupees and round

        st.sidebar.success(f"💰 Predicted Charges: **₹{prediction:,}**")
        st.sidebar.write(f"📈 Model Accuracy (R² Score): **{model_accuracy:.4f}**")

        # Save prediction
        existing_data = pd.read_csv(PREDICTIONS_FILE)
        new_data = pd.DataFrame([[
            age, sex, bmi, children, smoker, region, prediction
        ]], columns=["Age", "Sex", "BMI", "Children", "Smoker", "Region", "Predicted Charges"])

        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(PREDICTIONS_FILE, index=False)

        st.sidebar.write("✅ Prediction saved successfully!")

    except Exception as e:
        st.sidebar.error(f"❌ Prediction failed: {e}")

# Data exploration section
st.header("📊 Data Exploration")

if os.path.exists(PREDICTIONS_FILE):
    saved_data = pd.read_csv(PREDICTIONS_FILE)
    st.write("### 🔍 Saved Predictions")
    st.dataframe(saved_data)

    st.write("### 📈 Distribution of Predicted Charges")
    st.bar_chart(saved_data["Predicted Charges"])
else:
    st.warning("⚠️ No saved predictions found.")

# ROC Curve section
st.write("### 🧪 ROC Curve (Binary Classification of Charges: High vs Low)")

threshold = y_test.median()
y_true_binary = (y_test > threshold).astype(int)
y_scores = model.predict(X_test)
y_pred_binary = (y_scores > threshold).astype(int)

# ROC Curve
fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots(figsize=(3, 3))
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# Confusion matrix
cm = confusion_matrix(y_true_binary, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "High"])
fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
st.pyplot(fig_cm)
