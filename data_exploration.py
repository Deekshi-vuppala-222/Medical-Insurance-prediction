import streamlit as st
import pandas as pd
import io  

st.set_page_config(page_title="Data Exploration")

st.title("📊 Data Exploration")

@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

data = load_data()

st.subheader("📄 Raw Dataset")
st.dataframe(data)

st.subheader("📈 Summary Statistics")
st.write(data.describe())

st.subheader("🔍 Column Information")


buffer = io.StringIO()  
data.info(buf=buffer)   
info_str = buffer.getvalue()  
st.text(info_str)  
