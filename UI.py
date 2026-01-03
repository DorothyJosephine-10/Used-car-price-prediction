import streamlit as st
import numpy as np
import pickle
import base64

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded.decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function
add_bg_from_local("Background-image.jpg")

with open("Car_price_artifacts.pkl", "rb") as file:
    artifacts = pickle.load(file)

model = artifacts["model"]
encoders = artifacts["encoders"]

st.set_page_config(layout="wide")

st.title("🚗 Used Car Price Prediction")

# UI inputs(Categorical)

left, right = st.columns([2, 3])  # right is bigger

with left:
    col1, col2 = st.columns(2)

    with col1:
        ft = st.selectbox("Fuel Type", encoders["ft"].classes_)
        bt = st.selectbox("Body Type", encoders["bt"].classes_)
        transmission = st.selectbox("Transmission", encoders["transmission"].classes_)
        oem = st.selectbox("OEM", encoders["oem"].classes_)
        model_name = st.selectbox("Car Model", encoders["model"].classes_)
        color = st.selectbox("Color", encoders["Color"].classes_)

    with col2:
        city = st.selectbox("City", encoders["City"].classes_)
        km = st.number_input("Kilometers Driven", min_value=0.0)
        ownerNo = st.number_input("Owner Number", min_value=1, step=1)
        modelYear = st.number_input("Model Year", min_value=2000, max_value=2025)
        Seats = st.number_input("Seats", min_value=2, max_value=8)
        Mileage = st.number_input("Mileage (km/l)", min_value=0.0)
        
with right:
    st.image(
        "Car_image.png",
        width="stretch"
    )

# Encode categorical values (USING SAVED ENCODERS)

ft_enc = encoders["ft"].transform([ft])[0]
bt_enc = encoders["bt"].transform([bt])[0]
trans_enc = encoders["transmission"].transform([transmission])[0]
oem_enc = encoders["oem"].transform([oem])[0]
model_enc = encoders["model"].transform([model_name])[0]
color_enc = encoders["Color"].transform([color])[0]
city_enc = encoders["City"].transform([city])[0]

 # Arrange input in TRAINING ORDER (CRITICAL)

input_data = np.array([
    ft_enc,
    bt_enc,
    km,
    trans_enc,
    ownerNo,
    oem_enc,
    model_enc,
    modelYear,
    Seats,
    Mileage,
    color_enc,
    city_enc
]).reshape(1, -1)              
   
btn_col,btn_col_1 = st.columns([2, 3])

with btn_col:
    col3, col4 = st.columns([1, 3])

    with col3:
        predict = st.button("Predict Price")

    with col4:
        if predict:
            log_price = model.predict(input_data)
            price = np.expm1(log_price)

            st.success(f"💰 Predicted Car Price: ₹ {round(price[0], 2)} lakhs")
            st.balloons()




