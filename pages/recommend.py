# import module
import streamlit as st
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# from selenium import webdriver
from bs4 import BeautifulSoup
import re
from collections import Counter
import requests
from colorthief import ColorThief

st.set_page_config(page_title="Website Improvements", page_icon="⭐")
st.subheader("Issue Scanning")
image=st.file_uploader("upload your screenshot here")

def hex_to_numeric(hex_code):
  hex_code = hex_code.lstrip('#')
  r, g, b = int(hex_code[:2], 16), int(hex_code[2:4], 16), int(hex_code[4:], 16)
  return (r + g + b) / 3

if image:
  st.write("Image uploaded, majority two colours are")

  thief = ColorThief(image)
  palette = thief.get_palette(color_count=2)
  test_color_1 = '#%02x%02x%02x' % palette[1]
  test_color_2 = '#%02x%02x%02x' % palette[2]
  st.write('#%02x%02x%02x' % palette[1], '#%02x%02x%02x' % palette[2])

  st.text("")
  st.text("")
  st.text("")
  st.subheader("Dataset")
  # training the ai model :>
  #   link = "./BBCS-June-24/dataset/colour contrast ratio - color_contrast_data.csv"

  # creating the dataframe
  df = pd.read_csv("./dataset/colour contrast ratio - color_contrast_data.csv")


  # data clearning
  df['color1_numeric'] = df['Color1'].apply(hex_to_numeric)
  df['color2_numeric'] = df['Color2'].apply(hex_to_numeric)
  st.write(df.head())

  # training the AI ?
  from sklearn.model_selection import train_test_split
  x = df[["color1_numeric", "color2_numeric", "Contrast Ratio"]]
  y = df["Binary Scoring"]

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  # st.write(X_test)
  X_test = scaler.transform(X_test)

  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score

  clf = LogisticRegression(random_state=2)
  clf.fit(X_train, y_train)

  predictions = clf.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)


  import numpy as np
  st.text("")
  st.text("")
  st.text("")
  st.subheader("Evaluation")


  data = [[hex_to_numeric(test_color_1), hex_to_numeric(test_color_2), 2.5]]
  df = pd.DataFrame(data, columns=['color1_numeric', 'color2_numeric', 'Contrast Ratio'])
  prediction = clf.predict(df)
  result = (prediction[0] == 1)
  st.write("Result: "+str(prediction))
  if result:
    st.write("Meaning: Good Contrast")
  else:
    st.write("Meaning: Bad Contrast")


