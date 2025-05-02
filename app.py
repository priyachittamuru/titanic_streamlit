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
    st.image(titanic_image, caption='RMS Titanic (1912)', use_container_width=True)
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

# Simple binning logic (replace with your actual binning if different)
age_bin = np.digitize(age, bins=[0, 12, 18, 30, 50, 100]) - 1  # Creates bins 0-4
fare_bin = np.digitize(fare, bins=[0, 10, 20, 50, 100, 600]) - 1  # Creates bins 0-4

# Create feature vectors
embarked_features = {
    'C': [0, 1, 0, 0],  # [Embarked_, Embarked_C, Embarked_Q, Embarked_S]
    'Q': [0, 0, 1, 0],
    'S': [0, 0, 0, 1]
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
    # Create input array with EXACTLY 17 features in correct order:
    input_data = np.array([[
        pclass,                     # 1. Pclass
        0 if sex == "male" else 1,  # 2. Sex
        age,                        # 3. Age
        family_size,                # 4. FamilySize
        has_cabin,                  # 5. HasCabin
        is_alone,                   # 6. IsAlone
        age_bin,                    # 7. AgeBin
        fare_bin,                   # 8. FareBin
        *embarked_features[embarked],  # 9-12. Embarked_, Embarked_C, Embarked_Q, Embarked_S
        *title_features[title]      # 13-17. Title_Master, Title_Miss, Title_Mr, Title_Mrs, Title_Rare
    ]])
    
    # Debug output (can be removed after verification)
    st.write(f"Input shape: {input_data.shape} (should be (1, 17))")
    
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"✅ Survived (Probability: {probability:.2%})")
            st.balloons()
        else:
            st.error(f"❌ Did not survive (Probability: {probability:.2%})")
            
        # Show feature importance (if available)
        try:
            if hasattr(model, 'coef_'):
                st.subheader("Feature Importance")
                features = [
                    'Pclass', 'Sex', 'Age', 'FamilySize', 'HasCabin', 'IsAlone', 'AgeBin', 'FareBin',
                    'Embarked_', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
                    'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare'
                ]
                importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.coef_[0]
                }).sort_values('Importance', ascending=False)
                st.dataframe(importance)
        except:
            pass
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Please check that all input values are valid")

# Add some footer information
st.markdown("---")
st.markdown("""
**Note:** This prediction is based on a machine learning model trained on historical Titanic passenger data.
Actual outcomes might have varied due to unpredictable circumstances.
""")
