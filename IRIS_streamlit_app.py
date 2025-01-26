
import streamlit as st
import pickle

st.title("Iris_flower_Classification")
# Load the saved model
loaded_model = pickle.load(open('/content/logistic_model.sav','rb'))

# Create input fields for the four variables
SepalLengthCm = st.number_input("Enter Sepal Length (cm):")
SepalWidthCm = st.number_input("Enter Sepal Width (cm):")
PetalLengthCm = st.number_input("Enter Petal Length (cm):")
PetalWidthCm = st.number_input("Enter Petal Width (cm):")

# Create a button to trigger prediction
if st.button("Predict"):
    # Make prediction using the loaded model
    output = loaded_model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])

    # Display the prediction
    st.write(f"The predicted output is: {output[0]}")
