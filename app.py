
import streamlit as st

st.set_page_config(page_title="Laptop Price Prediction")

st.title("Laptop Price Prediction")
st.write("My Machine Learning Project")

st.success("Streamlit is working successfully!")

st.subheader("Enter Laptop Details")

ram = st.selectbox("RAM", [4, 8, 16, 32, 64])
storage = st.selectbox("Storage (GB)", [256, 512, 1024, 2048])
weight = st.number_input("Laptop Weight (kg)", 0.5, 5.0, 1.5)

st.write("RAM:", ram, "GB")
st.write("Storage:", storage, "GB")
st.write("Weight:", weight, "kg")
