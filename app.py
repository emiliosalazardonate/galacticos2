import streamlit as st

# Title and text
st.title("Hello Galaxy Anotator!")
st.write("This is a simple Streamlit app.")

# Input widget
name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello, {name}!")

# Slider example
age = st.slider("How confident are you of the galaxy:", 0, 100, 25)
st.write(f"Your age is {age}")