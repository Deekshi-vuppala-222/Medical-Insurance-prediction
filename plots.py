import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Visualizations")

st.title("📉 Data Visualizations")

@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

data = load_data()


st.subheader("📄 Dataset Overview")
st.dataframe(data.head())


st.sidebar.header("🔍 Filter Data")


age_options = sorted(data["age"].unique())
selected_age = st.sidebar.selectbox("Select Age", ["All"] + list(age_options))


children_options = sorted(data["children"].unique())
selected_children = st.sidebar.selectbox("Select Number of Children", ["All"] + list(children_options))


bmi_bins = ["All", "Underweight (<18.5)", "Normal (18.5-24.9)", "Overweight (25-29.9)", "Obese (≥30)"]
selected_bmi = st.sidebar.selectbox("Select BMI Category", bmi_bins)


smoker_options = data["smoker"].unique()
selected_smoker = st.sidebar.selectbox("Select Smoker Status", ["All"] + list(smoker_options))

region_options = data["region"].unique()
selected_region = st.sidebar.selectbox("Select Region", ["All"] + list(region_options))


sex_options = data["sex"].unique()
selected_sex = st.sidebar.selectbox("Select Sex", ["All"] + list(sex_options))


filtered_data = data.copy()

if selected_age != "All":
    filtered_data = filtered_data[filtered_data["age"] == selected_age]
if selected_children != "All":
    filtered_data = filtered_data[filtered_data["children"] == selected_children]
if selected_bmi != "All":
    if selected_bmi == "Underweight (<18.5)":
        filtered_data = filtered_data[filtered_data["bmi"] < 18.5]
    elif selected_bmi == "Normal (18.5-24.9)":
        filtered_data = filtered_data[(filtered_data["bmi"] >= 18.5) & (filtered_data["bmi"] <= 24.9)]
    elif selected_bmi == "Overweight (25-29.9)":
        filtered_data = filtered_data[(filtered_data["bmi"] >= 25) & (filtered_data["bmi"] <= 29.9)]
    elif selected_bmi == "Obese (≥30)":
        filtered_data = filtered_data[filtered_data["bmi"] >= 30]
if selected_smoker != "All":
    filtered_data = filtered_data[filtered_data["smoker"] == selected_smoker]
if selected_region != "All":
    filtered_data = filtered_data[filtered_data["region"] == selected_region]
if selected_sex != "All":
    filtered_data = filtered_data[filtered_data["sex"] == selected_sex]


if filtered_data.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
else:
    
    st.subheader("💰 Distribution of Insurance Charges")
    if st.sidebar.checkbox("Show Charges Distribution", True):
        fig1 = px.histogram(filtered_data, x="charges", nbins=40, title="Charges Distribution", color_discrete_sequence=["#3498db"])
        st.plotly_chart(fig1, use_container_width=True)

    
    st.subheader("📊 Age vs Insurance Charges")
    if st.sidebar.checkbox("Show Age vs Charges Scatter Plot", True):
        fig2 = px.scatter(filtered_data, x="age", y="charges", color="smoker", title="Age vs Charges", trendline="ols")
        st.plotly_chart(fig2, use_container_width=True)

    
    st.subheader("🚬 Smoker vs Non-Smoker Charges")
    if st.sidebar.checkbox("Show Smoker vs Non-Smoker Charges", True):
        fig3 = px.box(filtered_data, x="smoker", y="charges", color="smoker", title="Smoker vs Non-Smoker Charges")
        st.plotly_chart(fig3, use_container_width=True)

   
    st.subheader("📦 BMI vs Insurance Charges")
    if st.sidebar.checkbox("Show BMI vs Charges Box Plot", True):
        fig4 = px.box(filtered_data, x="bmi", y="charges", color="smoker", title="BMI vs Charges")
        st.plotly_chart(fig4, use_container_width=True)

    
    st.subheader("📍 Region-wise Insurance Charges")
    if st.sidebar.checkbox("Show Region-wise Charges Distribution", True):
        fig5 = px.box(filtered_data, x="region", y="charges", color="region", title="Charges by Region")
        st.plotly_chart(fig5, use_container_width=True)

    
    st.subheader("📈 Correlation Between Features")
    selected_features = st.sidebar.multiselect("Select Features for Correlation Heatmap", filtered_data.select_dtypes(include=['number']).columns, default=filtered_data.select_dtypes(include=['number']).columns)

    if selected_features:
        fig6, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(filtered_data[selected_features].corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig6)
    else:
        st.warning("Please select at least one feature for the correlation heatmap.")