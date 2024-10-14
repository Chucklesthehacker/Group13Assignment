import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import necessary libraries for machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

import joblib

gold_data = pd.read_csv('FINAL_USO.csv')
def clean_dataset(data):
    # Step 1: convert columns to datetime where possible, keep original value
    for col in data.select_dtypes(include=['object']).columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except:
            data[col] = data[col]

    for col in data.select_dtypes(include=['object']).columns:
        try:
            data[col] = pd.to_datetime(data[col])
        except:
            data[col] = data[col]

    # if col in data.columns.all() == 0:
    #     data[col] = data[col].astype('bool')
    # elif col in data.columns.all()  == 1:
    #     data[col] = data[col].astype('bool')
    # else:
    #     data[col] = data[col]

st.sidebar.title('Select Dataset(s)')
uploaded_file = st.sidebar.file_uploader("Choose one or more CSV files", type=['csv'], accept_multiple_files=True)

datasets = {}

if uploaded_file:
    for uploaded_files in uploaded_file:
        datasets[uploaded_file.name] = pd.read_csv(uploaded_file)

    selected_dataset = st.sidebar.selectbox('Select Dataset', options=list(datasets.keys()))

    data = datasets[selected_dataset]
    cleaned_data = data.drop_duplicates(keep='first')


# gold_data.drop_duplicates(keep='first')
# print(gold_data.shape)