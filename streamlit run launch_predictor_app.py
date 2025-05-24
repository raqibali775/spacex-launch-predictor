import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Train model
@st.cache_resource
def train_model():
    df = pd.read_csv("spacex_launch_data.csv")
    df = df[df['success'].notnull()]
    df['success'] = df['success'].astype(int)
    df['payload_count'] = df['payloads'].apply(lambda x: len(str(x).split(',')))
    X = df[['payload_count']]
    y = df['success']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

# Streamlit UI
st.title("🚀 SpaceX Launch Success Predictor")

payload_input = st.slider("Enter Payload Count", min_value=1, max_value=10, value=2)

if st.button("Predict"):
    prediction = model.predict([[payload_input]])
    if prediction[0] == 1:
        st.success("✅ Launch Likely to be SUCCESSFUL!")
    else:
        st.error("❌ Launch Might FAIL.")