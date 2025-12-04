import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Read data from CSV in the `data/` folder
data_path = os.path.join("data", "cleaned_properties dataset.csv")
df = pd.read_csv(data_path)
st.set_page_config(layout='wide')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

st.title("Smart Property Investment Advisor")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)


