import numpy as np
import pandas as pd
import seaborn as sms
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import datetime

df = pd.read_csv("/home/miracle/miracle_repo/MLOps/cars24-car-price.csv")
print(df.head(2))

with open("car_pred.pkl", "rb") as file:
    model = pickle.load(file)

st.header("Cars 24 prediction")
user_date = st.date_input("Select your Date",
                            value = datetime.date(2000, 6, 12),
                            min_value = datetime.date(2000, 1, 12),
                            max_value = datetime.date(2004, 1, 12)
                            )
# purchaseDate = st.date_input(label="purchase Date", value=datetime.date[2024,1,1], min_value=datetime.date(2000,1,1), max_value=(2024,1,1))