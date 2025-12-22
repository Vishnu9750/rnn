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
user_input_str = st.text_area("Last 30 Days Consumption (kWh):", value=", ".join(map(str, default_input_values)))

if st.button("Predict Next Day's Consumption"):    
    try:
        # Convert input string to a list of floats
        input_list = [float(x.strip()) for x in user_input_str.split(',') if x.strip()]

        if len(input_list) != n_steps:
            st.error(f"Please enter exactly {n_steps} consumption values.")
        else:
            # Convert to numpy array and reshape for scaling
            last_n_days_unscaled = np.array(input_list).reshape(-1, 1)

            # Scale the input data using the loaded scaler
            last_n_days_scaled = scaler.transform(last_n_days_unscaled)

            # Reshape for the RNN model (batch_size, n_steps, features)
            # For a single prediction, batch_size is 1
            model_input = last_n_days_scaled.reshape(1, n_steps, 1).astype(np.float32)

            # Make prediction
            predicted_scaled = model.predict(model_input, verbose=0)

            # Inverse transform the prediction to get original scale
            predicted_consumption = scaler.inverse_transform(predicted_scaled)[0][0]

            st.success(f"Predicted energy consumption for the next day: {predicted_consumption:.2f} kWh")

    except ValueError:
        st.error("Invalid input. Please ensure all values are numbers separated by commas.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
