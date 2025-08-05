# utils.py
import streamlit as st

def get_user_data(prompt_message):
    return st.file_uploader(prompt_message)