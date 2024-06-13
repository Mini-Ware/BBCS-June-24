# import module
import streamlit as st
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from selenium import webdriver
from bs4 import BeautifulSoup
import re
from collections import Counter
import requests
from colorthief import ColorThief

image=st.file_uploader("upload your screenshot here")



def get_dominant_colors():
  thief = ColorThief(image)
  palette = thief.get_palette(color_count=2)
  return palette


if image:
  get_dominant_colors(image)
  st.write(type(palette))
# training the ai model :>
  link = "./BBCS-June-24/dataset/colour contrast ratio - color_contrast_data.csv"

# creating the dataframe
  df = pd.read_csv(link)
  df.head()


# training the AI ?

  X = df["Color1", "Color2", "Contrast Ratio"]
  y = df["Binary Scoring"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  clf = LogisticRegression(random_state=2)

  clf.fit(X_train, y_train)

  predictions = clf.predict(X_test)

  accuracy = accuracy_score(y_test, predictions)
  st.write(predictions)
  st.write("Accuracy", accuracy)
