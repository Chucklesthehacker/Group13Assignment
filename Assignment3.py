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
#  A high strength correlation matrix
#  A genralized way to filter out boolean data

# correlation strength variables
high = 0.95
low =0.75


gold_data = pd.read_csv('FINAL_USO.csv')

# A function to clean the data by converting object columns to suitable data types
def clean_dataset(data):
    # attempt to convert columns to numeric where possible, keep original value if not
    # for col in data.select_dtypes(include=['object']).columns:
    #     try:
    #         data[col] = pd.to_numeric(data[col])
    #     except ValueError:
    #         data[col] = data[col].astype('object')
    #     except:
    #         pass

    # attempt to convert all remaining data types into datetime, and if not keep original value
    # for col in data.select_dtypes(exclude=['int64', 'float64']).columns:
    #     try:
    #         data[col] = pd.to_datetime(data[col])
    #     except ValueError:
    #         data[col] = data[col]

    data['Date'] = pd.to_datetime(data['Date'])
    bool_variables = ['EU_Trend', 'OF_Trend','OS_Trend', 'SF_Trend', 'USB_Trend', 'PLT_Trend', 'PLD_Trend', 'USDI_Trend']
    for variable in bool_variables:
        data[variable] = data[variable].astype('bool')
    return data


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

    # Removing unnecessary columns
    # st.write("### Remove extra columns")
    # keeping_columns = st.multiselect("Select columns to keep:", cleaned_data.columns)
    # count = 0
    # for col in cleaned_data.columns:
    #     if col not in keeping_columns:
    #         cleaned_data = cleaned_data.drop(columns=[col])
    #         count = count+1
    # st.write(f"### Removed extra {count} variables")
    # st.write("Remaining Variables: ", cleaned_data.columns.tolist())
    # 
    st.write('### Outlier Analysis: ')

    numeric_columns = cleaned_data.select_dtypes(include=['float64', 'int64'])

    if not numeric_columns.empty:
        Q1 = numeric_columns.quantile(0.25)
        Q3 = numeric_columns.quantile(0.75)
        IQR = Q3 - Q1

        outliers = (numeric_columns < (Q1-1.5*IQR)) | (numeric_columns > (Q3+1.5*IQR))

        outliers_count = outliers.sum()

        st.write("### Number of Outliers for each numeric Variable")
        dtype_df_outlier = pd.DataFrame(outliers_count, columns=["Number of Outliers"]).reset_index()
        dtype_df_outlier = dtype_df_outlier.rename(columns={"index": "Column Name"})
        st.dataframe(dtype_df_outlier, use_container_width=True)
    else:
        st.write("No appropriate data for analysis")

    # Missing Values
    st.write("### Step 7: Missing Values")
    missing_values = cleaned_data.isnull().sum()
    st.write("### Missing Values in each Variable")
    dtype_df_missing = pd.DataFrame(missing_values, columns=["Missing Values"]).reset_index()
    dtype_df_missing = dtype_df_missing.rename(columns={"index": "Column Name"})
    st.dataframe(dtype_df_missing, use_container_width=True)

    if not numeric_columns.empty:
        correlation_matrix = numeric_columns.corr()

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, ax=ax, cmap="coolwarm")
        st.pyplot(fig)

        target_correlations = correlation_matrix[target_variable].drop(target_variable)

        st.write(f"### Correlation of Features with Target Variable: '{target_variable}'")
        
        strong_corr = target_correlations[target_correlations.abs() >= high]
        moderate_corr = target_correlations[(target_correlations.abs() >= low) & (target_correlations.abs() < high)]
        weak_corr = target_correlations[target_correlations.abs() < low]
        
        st.write(f"### Strong Correlations (|correlation| >= {high})")
        st.dataframe(strong_corr, use_container_width=True)

        st.write(f"### Moderate Correlations ({low} <=|correlation| < {high})")
        st.dataframe(moderate_corr, use_container_width=True)

        st.write(f"### Weak Correlations (|correlation| <{low})")
        st.dataframe(weak_corr, use_container_width=True)

        # make new correlation heatmap for high correlation values ( make correlations great again)
        # high_corr_columns = correlation_matrix.columns
        # for column in correlation_matrix:
        #     high_correlation = correlation_matrix[column].drop(columns=[column])
        # fig, ax = plt.subplots(figsize=(14, 10))
        # sns.heatmap(high_correlation, annot=True, ax=ax, cmap="coolwarm")
        # st.pyplot(fig)
    else:
        st.write("No continuous numeric data for analysis.")

    categorical_columns = cleaned_data.select_dtypes(exclude=['float64', 'int64'])

    if not categorical_columns.empty:
        selected_categorical = st.multiselect("Select categorical variables for ANOVA: ", categorical_columns.columns)

        if pd.api.types.is_numeric_dtype(cleaned_data[target_variable]):
            anova_results = []

            for cat_col in selected_categorical:
                anova_groups = cleaned_data.groupby(cat_col)[target_variable].apply(list)
                f_val, p_val = stats.f_oneway(*anova_groups)

                anova_results.append({"Categorical Variable": cat_col, "F-Value": f_val, "P-Value": p_val})

            anova_df = pd.DataFrame(anova_results)

            st.write("### ANOVA Results")
            st.write("The following table shows F and P-values for each categorical variable.")
            st.dataframe(anova_df, use_container_width=True)

            if "P-Value" in anova_df.columns:
                significant_vars = anova_df[anova_df["P-Value"] < 0.05]
                st.write("### Significant Variables (P < 0.05):")
                if not significant_vars.empty:
                    st.dataframe(significant_vars, use_container_width=True)
                else:
                    st.write("No significant variables (P < 0.05) detected")

                st.write("### Box Plot: Categorical Variable vs Target Variable")
                for cat_col in selected_categorical:
                    fig, ax = plt.subplots(figsize=(14, 10))
                    sns.boxplot(x=cat_col,y=target_variable, data=cleaned_data, ax=ax)
                    ax.set_title(f'{cat_col} vs {target_variable}')
                    st.pyplot(fig)