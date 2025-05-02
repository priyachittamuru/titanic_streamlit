import streamlit as st
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO

# Load the model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Titanic Survival Prediction")
st.write("⚓ This app predicts whether a passenger would have survived the Titanic disaster.")

# Add Titanic image from web
try:
    response = requests.get('https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg')
    titanic_image = Image.open(BytesIO(response.content))
    st.image(titanic_image, caption='RMS Titanic (1912)', use_column_width=True)
except:
    st.warning("Couldn't load the Titanic image, proceeding without it")

# Collect user input
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3])
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    family_size = st.number_input("Family Size (Siblings/Spouses + Parents/Children)", 
                                min_value=0, max_value=10, value=0)
    has_cabin = st.selectbox("Has Cabin", options=[0, 1], 
                           format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    fare = st.number_input("Fare (£)", min_value=0.0, value=32.20)
    embarked = st.selectbox("Embarked", options=["C", "Q", "S"])
    title = st.selectbox("Title", options=["Mr", "Mrs", "Miss", "Master", "Rare"])

# Preprocess inputs
is_alone = 1 if family_size == 0 else 0

# Create feature vector in the same order as the model expects
embarked_features = {
    'C': [1, 0, 0],
    'Q': [0, 1, 0],
    'S': [0, 0, 1]
}

title_features = {
    'Master': [1, 0, 0, 0, 0],
    'Miss': [0, 1, 0, 0, 0],
    'Mr': [0, 0, 1, 0, 0],
    'Mrs': [0, 0, 0, 1, 0],
    'Rare': [0, 0, 0, 0, 1]
}

# Button to predict
if st.button("Predict Survival"):
    # Create input array in the exact order the model expects
    input_data = np.array([[
        pclass,
        0 if sex == "male" else 1,
        age,
        family_size,
        has_cabin,
        is_alone,
        0,  # AgeBin - would need calculation in a real app
        0,  # FareBin - would need calculation in a real app
        *embarked_features[embarked],
        *title_features[title]
    ]])
    
    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Survived (Probability: {probability:.2%})")
        st.balloons()
    else:
        st.error(f"❌ Did not survive (Probability: {probability:.2%})")

# Add some footer information
st.markdown("---")
st.markdown("""
**Note:** This prediction is based on a machine learning model trained on historical Titanic passenger data.
The actual outcome might have been different due to various unpredictable factors.
""")
