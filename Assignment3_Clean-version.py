import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import io

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

# Correlation Strength Variables
high = 0.95
low = 0.75

bool_variables = ['EU_Trend', 'OF_Trend', 'OS_Trend', 'SF_Trend', 'USB_Trend',
                      'PLT_Trend', 'PLD_Trend', 'USDI_Trend']

gold_data = pd.read_csv('FINAL_USO.csv')


# Creating a Cache to store the models to improve performance
@st.cache_resource
def train_models(X_train_scaled, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "KNN Regressor": KNeighborsRegressor(),
        "SVM Regressor": SVR()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    return trained_models


@st.cache_resource
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


# Creating a function to clean our dataset
def convert_variables(data, bool_v):
    for variable in bool_v:
        data[variable] = data[variable].astype('bool')

    if 'Date' in data.columns: # using an if so to not throw exception if date column doesn't exist
        data = data.set_index('Date')
    data = data.drop_duplicates()
    return data

# I couldn't find how to plot multiple bar charts, so have taken the example provided
def multiple_bars(data, variables):

    # Generating multiple subplots
    fig, subPlot = plt.subplots(nrows=1, ncols=len(variables), figsize=(20, 10))
    fig.suptitle('Bar charts of: ' + str(variables))

    for colName, plotNumber in zip(variables, range(len(variables))):
        data.groupby(colName).size().plot(kind='bar', ax=subPlot[plotNumber])

    st.pyplot(fig, use_container_width=True)

st.title('Using Machine Learning to calculate the closing price of gold')


st.write('# Step 1: Reading data with Python')

st.write('The first step is to load the dataset into Python, converting the columns to '
         'suitable datatypes and displaying a sample of the data, alongside what datatype each column is converted to.')
cleaned_columns = convert_variables(gold_data, bool_variables)
st.write('#### Shape before cleaning', gold_data.shape)
st.write('#### Shape after cleaning', cleaned_columns.shape)


col1, col2 = st.columns(2)
with col1:
    unclean_dtype_df = pd.DataFrame(gold_data.dtypes, columns=["data type"]).reset_index()
    unclean_dtype_df = unclean_dtype_df.rename(columns={"index": "Column Name"})
    st.dataframe(unclean_dtype_df)
with col2:
    clean_dtype_df = pd.DataFrame(cleaned_columns.dtypes, columns=["data type"]).reset_index()
    clean_dtype_df = clean_dtype_df.rename(columns={"index": "Column Name"})
    st.dataframe(clean_dtype_df)

st.write('List of variables in dataset:', gold_data.columns.tolist())


st.write('### Key observations after step 1:')

st.write('The sample data we have chosen contains 1718 rows of data for 81 columns (variables), including the open, '
         'close, high, low, and volume of Gold, Silver, Platinum, Rhodium and Palladium stock prices.')
st.write('Of the variables within the dataset, the key ones are outlined below:')
st.write('- Date: the date the stock price of each record was collected')
st.write('- High: the highest price of gold for a specific day')
st.write('- Low: the lowest price of gold for a specific day')
st.write('- Volume: the final price of gold for a specific day, and')
st.write('- Adj Close: The final price of Gold, adjusted for '
         'the deduction of dividends and other expenses of the price')
st.write('The variables above are repeated for the other commodities outlined'
         ' above, and as such we will not outline them individually. ')
st.write('As part of cleaning the data, duplicate rows were removed. '
         'It is noted that the dataset does not contain any repeated rows.')
st.divider()

st.write('# Step 2: Problem Statement Definition')
st.write('The purpose of the model is to compare our target variables (close price and adjusted close price) '
         'to other variables within the dataset, with the intention of identifying correlations for the purpose of '
         'making predictions of trends in our target variables.')
st.divider()


st.write('# Step 3: Visualizing the target variable and its distribution')

st.write("If the target variable’s distribution is skewed, then the predictive model will lead to poor results.")
st.write("To ensure accurate results, a bell curve is ideal, however a positive or "
         "negative skew are fine if not significant.")
st.write("As we have two potential target variables, Close and AdjClose, we will visualise them both")


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
ax[0].hist(cleaned_columns['Close'], bins=50, edgecolor='black',alpha=0.5)
ax[0].set_title("Distribution of Close")
ax[0].set_xlabel("Close")
ax[0].set_ylabel("Frequency")

ax[1].hist(cleaned_columns['Adj Close'], bins=50, edgecolor='black',alpha=0.5)
ax[1].set_title("Distribution of Adj Close")
ax[1].set_xlabel("Adj Close")
ax[1].set_ylabel("Frequency")

st.pyplot(fig)


st.write('### Observations from Step 3:')

st.write('Having a look at our histograms of our potential target variables we can see that they are both positively '
         'skewed, however there is enough of a bell curve to ensure accurate results. Additionally, it appears they are'
         'identical so we will asses their correlation.')
st.write('If it is 1, we can assume the variables are identical and can exclude one from further analysis')
st.write('Correlation of Close and AdjClose: ',cleaned_columns['Close'].corr(cleaned_columns['Adj Close']))
st.write('Given the correlation is 1, we will exclude Close from further analysis')
st.divider()


st.write('# Step 4: Basic Exploratory Data Analysis (EDA)')

st.write("This step is performed to gauge the overall shape of the data, i.e. the volume of data and "
         "columns present in the dataset.")
st.write('This step will begin the process of identifying whether particular columns are kept, and provides a mechanism'
         ' for identifying data that is less impactful on the outcome.')
st.write('To perform this, there are 4 pandas methods/attributes used to analyze the data,'
         ' these are describe(), dtypes, info() and nunuique()')
st.write('- nunique(): this method is used to display the unique values within each column, '
         'the more unique values within a column the more useful for data analysis. '
         'It can be assumed that a column with less than 20 values can be classed as categorical, '
         'and we will investigate these later.')
st.write('- dtypes: this attribute is used to define the type of data in the column, '
         'this was is to ensure that data that was still useful in our data set was not used in comparisons '
         'that would not be appropriate. This will primarily be used to differentiate the variables with Boolean values'
         ' from the other numeric variables in the data set.')
st.write('- info: this method was used to primarily display how many non-null values were present in the dataset.')
st.write('- describe(): this method is used to provide an overview of each variable’s descriptive statistics, '
         'including the count of rows, mean, min, 25th, 50th and 75th percentiles, max and the standard deviation')


# Creating 2 columns for data types and summary
col1, col2 = st.columns(2)

with col1:
    st.write(f"### Data Types: (dtypes attribute)")
    dtype_df = pd.DataFrame(cleaned_columns.dtypes, columns=["data type"]).reset_index()
    dtype_df = dtype_df.rename(columns={"index": "Column Name"})

    st.dataframe(dtype_df, use_container_width=True)

with col2:
    st.write("### Unique Values (nunique() method)")
    st.dataframe(cleaned_columns.nunique(), use_container_width=True)

st.write("### Summary Statistics: (describe() method)")
st.dataframe(cleaned_columns.describe(), use_container_width=True)

# Creating a buffer to pipe the output of info() method to show in streamlit
buffer = io.StringIO()
cleaned_columns.info(buf=buffer)
cleaned_data_info = buffer.getvalue()
st.write("### Description of Dataset (info() method)")
st.text(cleaned_data_info)

st.write("### Observations from Step 4: ")
st.write("From the output of these methods and attributes, we can ascertain that there are:")
st.write("- 8 boolean variables, and")
st.write("- 72 continuous variables")
st.write("Additionally, as shown in the info() method, there are no null values present, with each variable"
         " containing 1718 entries")
st.write("From the nunique() method, we can see that there is a large variation of entries for each variable, "
         "excluding boolean which by nature has 2, which shows that at this stage there "
         "are many potential predictor variables.")
st.divider()


st.write("# Step 5 Visual EDA")
st.write("In this step we will visualize the distribution of all categorical predictor variables. "
         "That is any variable that has less than 20 values, and has repetition of "
         "values so that data can be grouped by the unique values")
st.write("From the basic analysis performed in the previous step, we identified 8 categorical predictors. These are:")
st.write(bool_variables)
st.write("We will use bar charts to show how the data is distributed in each variable")

# Calling the function defined at the start to plot the bar charts
multiple_bars(data=cleaned_columns, variables=bool_variables)

st.write("### Observations from Step 5: ")
st.write("The bar charts above show the frequency of each unique value in the variable. Ideally, each value within "
         "the variable will have comparable frequencies.")
st.write("As shown, all 8 categorical predictor variables have an almost perfectly even distribution, as such we "
         "will be able to use them all for the ML regression algorithm to learn.")
st.divider()

