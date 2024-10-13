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

gold_prices = pd.read_csv('FINAL_USO.csv')

def clean_dataset(data):
    # Step 1: convert columns to datetime where possible, keep original value
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_datetime(data[col])


clean_dataset(gold_prices)

gold_prices.info()