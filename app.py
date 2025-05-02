import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Load the model
@st.cache_resource
def load_model():
    with open('logistic_regression_model (1).pkl', 'rb') as f:
        return pickle.load(f)
model = load_model()

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

# Load Titanic image from Wikipedia
@st.cache_data
def load_image():
    response = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg")
    return Image.open(BytesIO(response.content))
titanic_img = load_image()

# App layout
st.title("Titanic Survival Predictor 🚢")
st.write("Predict whether a passenger would have survived the Titanic disaster using machine learning.")

# Create columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    # Display Titanic image with caption
    st.image(titanic_img, caption="RMS Titanic departing Southampton on April 10, 1912 (Source: Wikipedia)", 
             use_column_width=True)
    
    # Fun facts section
    with st.expander("💡 Titanic Facts"):
        st.write("""
        - Total passengers: 2,224
        - Survivors: 710 (32%)
        - First class survival rate: 63%
        - Third class survival rate: 25%
        - Women and children survival rate: 74%
        """)

with col2:
    # Passenger input form
    with st.form("passenger_form"):
        st.subheader("Passenger Information")
        
        # Create 3 columns for inputs
        left, middle, right = st.columns(3)
        
        with left:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], 
                                help="1 = First Class, 2 = Second Class, 3 = Third Class")
            sex = st.selectbox("Sex", ["female", "male"])
            age = st.slider("Age", 0.0, 100.0, 30.0)
            
        with middle:
            sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
            parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
            fare = st.number_input("Fare (£)", 0.0, 600.0, 32.0, step=1.0,
                                 help="Ticket fare in British pounds")
            
        with right:
            embarked = st.selectbox("Embarkation Port", 
                                  ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"],
                                  index=2)
            cabin = st.text_input("Cabin Number (if known)", 
                                help="Leave blank if unknown")
            
        # Prediction button
        submitted = st.form_submit_button("Predict Survival")
        
    # Prediction results
    if submitted:
        # Prepare features
        port_code = embarked[embarked.find("(")+1:embarked.find(")")]
        features = {
            'Pclass': pclass,
            'Sex': 1 if sex == "male" else 0,
            'Age': age,
            'FamilySize': sibsp + parch,
            'HasCabin': 1 if cabin else 0,
            'IsAlone': 1 if (sibsp + parch) == 0 else 0,
            'Embarked_C': 1 if port_code == "C" else 0,
            'Embarked_Q': 1 if port_code == "Q" else 0,
            'Embarked_S': 1 if port_code == "S" else 0,
            'Title_Mr': 1 if sex == "male" else 0,
            'Title_Miss': 1 if sex == "female" and age < 18 else 0,
            'Title_Mrs': 1 if sex == "female" and age >= 18 else 0,
            'Fare': fare,
            'AgeBin': 0,  # Placeholder - would need binning logic
            'FareBin': 0,  # Placeholder
            'Title_Master': 0,
            'Title_Rare': 0
        }
        
        # Create DataFrame and predict
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        
        # Display result with emojis
        result_container = st.container()
        with result_container:
            if prediction == 1:
                st.success("## 🎉 Survival Prediction: **Would Have Survived**")
                st.balloons()
            else:
                st.error("## 💀 Survival Prediction: **Would Not Have Survived**")
            
            # Show probability if available
            try:
                proba = model.predict_proba(df)[0][1]
                st.metric("Survival Probability", f"{proba:.1%}")
            except:
                pass

# Footer
st.markdown("---")
st.caption("""
Note: This predictive model is based on historical passenger data from 1912. 
Actual survival outcomes may have varied based on circumstances not captured in this model.
""")