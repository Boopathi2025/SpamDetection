import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit app interface
st.title("📧 Spam Detection App")

st.write("Enter a message below to check if it is **Spam** or **Not Spam**:")

# Input box
user_input = st.text_area("Message Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        # Vectorize input
        input_vec = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vec)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This message is **SPAM**.")
        else:
            st.success("✅ This message is **NOT SPAM**.")
