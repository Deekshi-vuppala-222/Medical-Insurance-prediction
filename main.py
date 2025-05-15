import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Insurance Charges Dashboard", layout="wide")

PREDICTIONS_FILE = "saved_predictions.csv"


if not os.path.exists(PREDICTIONS_FILE):
    pd.DataFrame(columns=["Age", "Sex", "BMI", "Children", "Smoker", "Predicted Charges"]).to_csv(PREDICTIONS_FILE, index=False)


page = st.sidebar.radio("📌 Navigation", ["🏠 Home", "📊 Data Exploration", "📈 Plots", "🔮 Prediction"])


if page == "🏠 Home":
    st.markdown('<h1 class="title">🏠 Insurance Charges Dashboard</h1>', unsafe_allow_html=True)
    st.write("""
        Welcome to the **Insurance Charges Prediction Dashboard**! 🚀  
        This application helps users **predict medical insurance charges** based on multiple factors.  
        
        ### 📌 Features of the App:
        - **📊 Data Exploration**: View all saved predictions.
        - **📈 Plots**: Visualize trends in predicted insurance charges.
        - **🔮 Prediction**: Enter details and get an estimated insurance cost.
        
        Try out the **Prediction** feature to generate an estimate!  
    """)

    if os.path.exists(PREDICTIONS_FILE):
        data = pd.read_csv(PREDICTIONS_FILE)
        if not data.empty:
            st.header("📌 Recent Predictions")
            st.dataframe(data.tail(5))  
        else:
            st.info("No predictions have been made yet. Try using the '🔮 Prediction' feature!")


elif page == "🔮 Prediction":
    st.markdown('<h1 class="title">🔮 Insurance Charges Prediction</h1>', unsafe_allow_html=True)
    
    st.write("""
        Enter your details below to get an estimated insurance charge.  
        **The prediction is based on age, BMI, children, and smoking status.**  
    """)

    
    age = st.number_input("Enter Age", min_value=18, max_value=100, step=1)
    sex = st.selectbox("Select Sex", ["Male", "Female"])
    bmi = st.number_input("Enter BMI", min_value=10.0, max_value=50.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
    smoker = st.selectbox("Smoker?", ["Yes", "No"])

    
    predicted_charges = age * bmi * (2 if smoker == "Yes" else 1)

    if st.button("🚀 Predict Charges"):
        st.success(f"💰 Predicted Insurance Charges: **${predicted_charges:.2f}**")

        
        existing_data = pd.read_csv(PREDICTIONS_FILE)
        new_data = pd.DataFrame([[age, sex, bmi, children, smoker, predicted_charges]],
                                columns=["Age", "Sex", "BMI", "Children", "Smoker", "Predicted Charges"])
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(PREDICTIONS_FILE, index=False)

        st.write("✅ Prediction saved successfully!")


elif page == "📊 Data Exploration":
    st.markdown('<h1 class="title">📊 Data Exploration</h1>', unsafe_allow_html=True)
    
    st.write("""
        Here, you can **view all previous insurance charge predictions** made using this app.  
        You can analyze how different factors (age, BMI, smoking) impact insurance costs.
    """)

    if os.path.exists(PREDICTIONS_FILE):
        data = pd.read_csv(PREDICTIONS_FILE)
        if not data.empty:
            st.write("### 🔍 Saved Predictions")
            st.dataframe(data)
        else:
            st.info("No saved predictions found. Try making a prediction first!")
    else:
        st.warning("⚠️ No data available.")


elif page == "📈 Plots":
    st.markdown('<h1 class="title">📈 Insurance Prediction Trends</h1>', unsafe_allow_html=True)

    st.write("""
        This section provides a **graphical analysis** of all saved insurance charge predictions.  
        You can observe trends and distributions in predicted costs.
    """)

    if os.path.exists(PREDICTIONS_FILE):
        data = pd.read_csv(PREDICTIONS_FILE)
        if not data.empty:
            st.write("### 📊 Distribution of Predicted Charges")
            st.bar_chart(data["Predicted Charges"])
        else:
            st.info("No data available for plotting. Try making some predictions first!")
    else:
        st.warning("⚠️ No data available for plotting.")
