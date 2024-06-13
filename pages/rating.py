# import module
import streamlit as st
import pandas as pd
import numpy as np

# rating to-do list
# svc classification model
# sentient analysis
# filter out positive comments

st.header("Community Rating")
df = pd.read_csv("./dataset/comments.csv")

st.subheader("Dataset")
st.write(df.head())
st.bar_chart(df.groupby(["category"]).size())

# for testing
# just use LLM to generate fake comments and ratings to train
# "give me another 20 short website review comments about x accessibility problems, i want this to be csv format with just one column and don't put semicolons"

st.text("")
st.text("")
st.text("")
st.subheader("Give Comment")
comment = st.text_input("Website improvements to reduce reliance on screen reader summariser?", value="there is a thing flying around the screen")



