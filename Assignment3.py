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

# Features to add:
#  Remove duplicate columns


gold_data = pd.read_csv('FINAL_USO.csv')

# A function to clean the data by converting object columns to suitable data types
def clean_dataset(data):
    # attempt to convert columns to numeric where possible, keep original value if not
    for col in data.select_dtypes(include=['object']).columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            data[col] = data[col].astype('object')
        except:
            pass

    # attempt to convert all remaining data types into datetime, and if not keep original value
    for col in data.select_dtypes(exclude=['int64', 'float64']).columns:
        try:
            data[col] = pd.to_datetime(data[col])
        except ValueError:
            data[col] = data[col]

    # attempt to convert numeric data types to boolean if applicable, otherwise keep original datatype
    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        try:
            data[col] = data[col].convert_dtypes()
        except ValueError:
            data[col] = data[col]

    return data

    # if col in data.columns.all() == 0:
    #     data[col] = data[col].astype('bool')
    # elif col in data.columns.all()  == 1:
    #     data[col] = data[col].astype('bool')
    # else:
    #     data[col] = data[col]

st.sidebar.title('Select Dataset(s)')
uploaded_files = st.sidebar.file_uploader("Choose one or more CSV files", type=['csv'], accept_multiple_files=True)

datasets = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        datasets[uploaded_file.name] = pd.read_csv(uploaded_file)

    selected_dataset = st.sidebar.selectbox('Select Dataset: ', options=list(datasets.keys()))

    data = datasets[selected_dataset]
    cleaned_data = data.drop_duplicates()

    # Convert object data types to numeric or datetime
    cleaned_data = clean_dataset(cleaned_data)

    st.title(f'Processing Dataset: {selected_dataset}')

    st.write('### Sample Data from the Selected Dataset')
    st.dataframe(cleaned_data.sample(5), use_container_width=True)

    # This is what Pranav used, so don't know if we need it or not. From what I can tell we don't need it with our dataset
    # if "Unnamed: 0" in cleaned_data.columns:
    #     cleaned_data = cleaned_data.drop(columns=["Unnamed: 0"])
    #     st.write("Dropped 'Unnamed: 0' column from the dataset.")

    # Show cleaned data
    st.write('### Cleaned data types')
    dtype_df = pd.DataFrame(cleaned_data.dtypes, columns=["data type"]).reset_index().rename(
        columns={"index": "Column Name"})
    st.dataframe(dtype_df, use_container_width=True)

    # Show shape and columns of selected dataset
    st.write('## Dataset Information')
    st.write(f'## Shape: {cleaned_data.shape}')
    st.write(f'Columns in the dataset: ', cleaned_data.columns.tolist())

    # Select the Target Variable
    target_variable = st.selectbox('Select the Target Variable:', options=list(cleaned_data.columns))
    st.write(f"### Target Variable '{target_variable}'")

    st.write("## Visualizing the Target Variable")

    fig, ax = plt.subplots()
    ax.hist(cleaned_data[target_variable], bins=50, edgecolor='black', alpha=0.5)
    ax.set_title(f'Distribution of {target_variable}')
    ax.set_xlabel(target_variable)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)



    # <>We need to figure out what kind of plots we need/want

    # comparison_columns = st.multiselect("Select Variables to compare: ", cleaned_data.columns)
    # if len(comparison_columns) > 1:
    #     sns.pairplot(cleaned_data[comparison_columns])
    #     st.pyplot(plt.gcf())

    # Creating 2 columns for data types and summary
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"### Data Types: ")
        dtype_df = pd.DataFrame(cleaned_data.dtypes, columns=["data type"]).reset_index()
        dtype_df = dtype_df.rename(columns={"index": "Column Name"})

        st.dataframe(dtype_df, use_container_width=True)

    with col2:
        st.write("### Summary Statistics:")
        st.dataframe(cleaned_data.describe(), use_container_width=True)

    # Visualising EDA - Histograms for continuous columns
    continuous_variables = st.multiselect("Select continuous variables to visualize: ",
                                          cleaned_data.select_dtypes(include=['float64', 'int64']).columns)

    if continuous_variables:
        for col in continuous_variables:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(cleaned_data[col], bins=50, edgecolor='black', alpha=0.5)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

    # This is what I'm working on ATM

    # Removing unnecessary columns
    st.write("### Remove extra columns")
    keeping_columns = st.multiselect("Select columns to keep:", cleaned_data.columns)
    count = 0
    for col in cleaned_data.columns:
        if col not in keeping_columns:
            cleaned_data = cleaned_data.drop(columns=[col])
            count = count+1
    st.write(f"### Removed extra {count} columns")
    st.write("### Remaining Variables: ")

    st.write(cleaned_data.columns.tolist())

    st.write('### Outlier Analysis: ')


    # numeric_columns = cleaned_data.select_dtypes(include=['float64', 'int64'])
    #
    # if not numeric_columns.empty:
    #     Q1 = cleaned_data[numeric_columns].quantile(0.25)
    #     Q3 = cleaned_data[numeric_columns].quantile(0.75)
    #     IQR = Q3 - Q1
