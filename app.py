import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("🏠 House Price Prediction App")

st.write("Enter house details to predict price")

area = st.number_input("Area (sqft)", 500, 10000, 3000)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
stories = st.number_input("Stories", 1, 5, 2)
parking = st.number_input("Parking", 0, 5, 2)

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

if st.button("Predict Price"):
    raw = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishing
    }

    binary_cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
    for col in binary_cols:
        raw[col] = 1 if raw[col] == "yes" else 0

    raw["furnishingstatus_semi-furnished"] = 1 if raw["furnishingstatus"] == "semi-furnished" else 0
    raw["furnishingstatus_unfurnished"] = 1 if raw["furnishingstatus"] == "unfurnished" else 0
    del raw["furnishingstatus"]

    df = pd.DataFrame([raw])
    df = df.reindex(columns=features, fill_value=0)

    df_scaled = pd.DataFrame(scaler.transform(df), columns=features)
    price = model.predict(df_scaled)[0]

    st.success(f"Predicted House Price: ₹{int(price):,}")
