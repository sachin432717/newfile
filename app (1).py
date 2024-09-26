import streamlit as st
import pandas as pd
import pickle

# Load the trained Ridge Regression model
filename = 'ensemble_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a title for your app
st.title('Revenue Prediction App')

# Create input fields for user to enter features
st.header('Enter Store Details')
average_order_value = st.number_input('Average Order Value')
total_orders = st.number_input('Total Orders')
customer_lifetime_value = st.number_input('Customer Lifetime Value')
average_customer_order_frequency = st.number_input('Average Customer Order Frequency')
monthly_operating_costs = st.number_input('Monthly Operating Costs')

# Create a button to make predictions
if st.button('Predict Monthly Revenue'):
  # Create a DataFrame with user input
  input_data = pd.DataFrame({
      'average_order_value': [average_order_value],
      'total_orders': [total_orders],
      'customer_lifetime_value': [customer_lifetime_value],
      'average_customer_order_frequency': [average_customer_order_frequency],
      'monthly_operating_costs': [monthly_operating_costs]
  })

  # Make predictions using the loaded model
  prediction = loaded_model.predict(input_data)[0]

  # Display the prediction
  st.header('Predicted Monthly Revenue')
  st.write(f'{prediction:.2f}')

