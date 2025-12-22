import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model('rnn_energy_model.keras')

# Load the fitted scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# n_steps should match the value used during model training
n_steps = 30

st.title("Energy Consumption Prediction App")
st.write("Predict the next day's energy consumption using an RNN model.")

# --- Input for the last 30 days consumption ---
st.header("Input Last 30 Days of Consumption")
st.write(f"Please enter the energy consumption (kWh) for the last {n_steps} days, separated by commas.")

# To provide a default value for demonstration purposes:
# First, read the original data to get the last n_steps values
# In a real deployment, this would likely come from an external source or database.
df_original = pd.read_csv("daily_energy_consumption.csv")
df_original['Date'] = pd.to_datetime(df_original['Date'], format="%d-%m-%Y")
df_original.set_index('Date', inplace=True)
consumption_data_original = pd.to_numeric(
    df_original['Consumption (kWh)'], errors='coerce'
).dropna()


default_input_values = consumption_data_original.iloc[-n_steps:].tolist()
st.write("Default values:", default_input_values)

user_input_str = st.text_area("Last 30 Days Consumption (kWh):", value=", ".join(map(str, default_input_values)))

if st.button("Predict Next Day's Consumption"):    
    try:
        # --- Parse input ---
        cleaned_input = user_input_str.replace("，", ",").replace("\n", "")
        input_list = [float(x.strip()) for x in cleaned_input.split(",") if x.strip()]

        # 🔍 DEBUG 1
        st.write("Parsed input length:", len(input_list))
        st.write("Parsed input sample:", input_list[:5])

        if len(input_list) != n_steps:
            st.error(f"Please enter exactly {n_steps} consumption values.")
        else:
            # --- Convert to numpy ---
            last_n_days_unscaled = np.array(input_list).reshape(-1, 1)

            # 🔍 DEBUG 2
            st.write("Unscaled shape:", last_n_days_unscaled.shape)

            # --- Scale ---
            last_n_days_scaled = scaler.transform(last_n_days_unscaled)

            # 🔍 DEBUG 3
            st.write("Scaled shape:", last_n_days_scaled.shape)

            # --- Reshape for model ---
            model_input = last_n_days_scaled.reshape(1, n_steps, 1).astype(np.float32)

            # 🔍 DEBUG 4
            st.write("Model input shape:", model_input.shape)

            # --- Predict ---
            predicted_scaled = model.predict(model_input, verbose=0)
            predicted_consumption = scaler.inverse_transform(predicted_scaled)[0][0]

            st.success(
                f"Predicted energy consumption for the next day: {predicted_consumption:.2f} kWh"
            )

    except Exception as e:
        st.error(f"REAL ERROR → {type(e).__name__}: {e}")



