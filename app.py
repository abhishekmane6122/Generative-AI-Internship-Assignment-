import streamlit as st
from chatbot import chatbot_response

st.title("Apple Vision Pro Chatbot")
user_input = st.text_input("You: ")

if user_input:
    response = chatbot_response(user_input)
    st.write(f"Bot: {response}")