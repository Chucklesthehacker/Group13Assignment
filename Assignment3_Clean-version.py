import streamlit
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


import joblib

# Correlation Strength Variables
high = 0.9
low = 0.75

bool_variables = ['EU_Trend', 'OF_Trend', 'OS_Trend', 'SF_Trend', 'USB_Trend',
                      'PLT_Trend', 'PLD_Trend', 'USDI_Trend']

gold_data = pd.read_csv('FINAL_USO.csv')

# Downloaded the strong correlation DF from streamlit and reading here to create a DF for later
strong_corr_df = pd.read_csv('high_corr.csv')

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
    fig, subPlot = plt.subplots(nrows=1, ncols=len(variables), figsize=(20, 5))
    fig.suptitle('Bar charts of: ' + str(variables))

    for colName, plotNumber in zip(variables, range(len(variables))):
        data.groupby(colName).size().plot(kind='bar', ax=subPlot[plotNumber])

    st.pyplot(fig, use_container_width=True)



st.title('Using Machine Learning to calculate the closing price of gold')


st.write('# Step 1: Reading data with Python')

st.write('The first step is to load the dataset into Python, converting the columns to '
         'suitable datatypes and displaying a sample of the data, alongside what datatype each column is converted to.')

cleaned_data = convert_variables(gold_data, bool_variables)
cleaned_data_copy = cleaned_data

st.write('#### Shape before cleaning', gold_data.shape)
st.write('#### Shape after cleaning', cleaned_data.shape)


col1, col2 = st.columns(2)
with col1:
    unclean_dtype_df = pd.DataFrame(gold_data.dtypes, columns=["data type"]).reset_index()
    unclean_dtype_df = unclean_dtype_df.rename(columns={"index": "Column Name"})
    st.dataframe(unclean_dtype_df)
with col2:
    clean_dtype_df = pd.DataFrame(cleaned_data.dtypes, columns=["data type"]).reset_index()
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
ax[0].hist(cleaned_data['Close'], bins=50, edgecolor='black', alpha=0.5)
ax[0].set_title("Distribution of Close")
ax[0].set_xlabel("Close")
ax[0].set_ylabel("Frequency")

ax[1].hist(cleaned_data['Adj Close'], bins=50, edgecolor='black', alpha=0.5)
ax[1].set_title("Distribution of Adj Close")
ax[1].set_xlabel("Adj Close")
ax[1].set_ylabel("Frequency")

st.pyplot(fig)


st.write('### Observations from Step 3:')

st.write('Having a look at our histograms of our potential target variables we can see that they are both positively '
         'skewed, however there is enough of a bell curve to ensure accurate results. Additionally, it appears they are'
         'identical so we will asses their correlation.')
st.write('If it is 1, we can assume the variables are identical and can exclude one from further analysis')
st.write('Correlation of Close and AdjClose: ', cleaned_data['Close'].corr(cleaned_data['Adj Close']))
st.write('Given the correlation is 1, we will exclude Close from further analysis')

cleaned_data = cleaned_data.drop(columns=['Close'])
cleaned_data_copy = cleaned_data_copy.drop(columns=['Close'])
target_variable = cleaned_data['Adj Close']
predictor_variables = cleaned_data.drop(columns=['Adj Close'])
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
    dtype_df = pd.DataFrame(cleaned_data.dtypes, columns=["data type"]).reset_index()
    dtype_df = dtype_df.rename(columns={"index": "Column Name"})

    st.dataframe(dtype_df, use_container_width=True)

with col2:
    st.write("### Unique Values (nunique() method)")
    st.dataframe(cleaned_data.nunique(), use_container_width=True)

st.write("### Summary Statistics: (describe() method)")
st.dataframe(cleaned_data.describe(), use_container_width=True)

# Creating a buffer to pipe the output of info() method to show in streamlit
buffer = io.StringIO()
cleaned_data.info(buf=buffer)
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


st.write("# Step 5 Visual EDA of Categorical Predictor Variables")
st.write("In this step we will visualize the distribution of all categorical predictor variables. "
         "That is any variable that has less than 20 values, and has repetition of "
         "values so that data can be grouped by the unique values")
st.write("From the basic analysis performed in the previous step, we identified 8 categorical predictors. These are:")
st.write(bool_variables)
st.write("We will use bar charts to show how the data is distributed in each variable")

# Calling the function defined at the start to plot the bar charts
multiple_bars(data=cleaned_data, variables=bool_variables)

st.write("### Observations from Step 5: ")
st.write("The bar charts above show the frequency of each unique value in the variable. Ideally, each value within "
         "the variable will have comparable frequencies.")
st.write("As shown, all 8 categorical predictor variables have an almost perfectly even distribution, as such we "
         "will be able to use them all for the ML regression algorithm to learn.")
st.divider()


st.write("# Step 6: Visual EDA of Continuous Predictor Variables")

continuous_variables = cleaned_data.select_dtypes(include=["float64", 'int64'])
continuous_variables = continuous_variables.drop(columns=['Adj Close'])

col_count = 0

for col in continuous_variables.columns:
    col_count +=1
st.write(f"From the basic EDA performed earlier, we know there are {col_count} continuous predictor"
         f" variables in the dataset ")
st.write("Given the volume of continuous predictor values, we will now make a decision on which variables to keep "
         "to perform further analysis on, and visualize these")
st.write("As we are looking to predict the price of gold, we will keep the 4 variables associated with gold ,'Open', "
         "'High', 'Low' and 'Adj Close'. Additionally, we will keep the variables relating to other precious metals, "
         "as they can potentially help the ML model in predicting gold price. We will also keep the categorical"
         " variables from the previous step in case we need them later on.")

variables_to_keep = ["PLT_Open", "PLT_Price","PLT_Low", "PLT_High","PLD_Open", "PLD_High", "PLD_Low",
                     "PLD_Price", "RHO_PRICE","Open", "High", "Low", "Adj Close"]
for variable in bool_variables:
    variables_to_keep.append(variable)

for col in cleaned_data.columns:
    if col not in variables_to_keep:
        cleaned_data = cleaned_data.drop(columns=[col])

for col in continuous_variables.columns:
    if col not in variables_to_keep:
        continuous_variables = continuous_variables.drop(columns=[col])
st.write("Continuous Predictor variables kept for analysis:",continuous_variables.columns.tolist())

fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].hist(cleaned_data['Open'], bins=30)
ax[0].set_title('Distribution of High')
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('High')

ax[1].hist(cleaned_data['Low'], bins=30)
ax[1].set_title('Distribution of Low')
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('Low')

ax[2].hist(cleaned_data['Open'], bins=30)
ax[2].set_title('Distribution of Open')
ax[2].set_ylabel('Frequency')
ax[2].set_xlabel('Open')
st.pyplot(fig)

fig, ax = plt.subplots(ncols=4, figsize=(20,5))

ax[0].hist(cleaned_data['PLT_Open'], bins=30)
ax[0].set_title('Distribution of PLT_Open')
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('PLT_Open')

ax[1].hist(cleaned_data['PLT_High'], bins=30)
ax[1].set_title('Distribution of PLT_High')
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('PLT_High')

ax[2].hist(cleaned_data['PLT_Low'], bins=30)
ax[2].set_title('Distribution of PLT_Low')
ax[2].set_ylabel('Frequency')
ax[2].set_xlabel('PLT_Low')

ax[3].hist(cleaned_data['PLT_Price'], bins=30)
ax[3].set_title('Distribution of PLT_Price')
ax[3].set_ylabel('Frequency')
ax[3].set_xlabel('PLT_Price')

st.pyplot(fig)

fig, ax = plt.subplots(ncols=4, figsize=(20,5))

ax[0].hist(cleaned_data['PLD_Price'], bins=30)
ax[0].set_title('Distribution of PLD_Price')
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('PLD_Price')

ax[1].hist(cleaned_data['PLD_Open'], bins=30)
ax[1].set_title('Distribution of PLD_Open')
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('PLD_Open')

ax[2].hist(cleaned_data['PLD_High'], bins=30)
ax[2].set_title('Distribution of PLD_High')
ax[2].set_ylabel('Frequency')
ax[2].set_xlabel('PLD_High')

ax[3].hist(cleaned_data['PLD_Low'], bins=30)
ax[3].set_title('Distribution of PLD_Low')
ax[3].set_ylabel('Frequency')
ax[3].set_xlabel('PLD_Low')

st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,5))

ax.hist(cleaned_data['RHO_PRICE'], bins=30)
ax.set_title('Distribution of RHO_Price')
ax.set_ylabel('Frequency')
ax.set_xlabel('RHO_PRICE')

st.pyplot(fig)

st.write("### Observations from Step 6: ")
st.write("As shown in the histograms, the first three predictor variables are positively skewed. They have a decent "
         "normal distribution, with a smaller secondary peak at higher values, and we can remove those outliers to "
         "assist in the ML prediction later on in the analysis if it's required. ")
st.write("The histograms for the variables relating to platinum show that the distribution is evenly skewed across two "
         "peaks at higher and lower values, so we will keep the distribution as is")
st.write("The distribution of palladium shows a fairly normalised bell curve, with a slight positive skew."
         " As such we will keep palladium unchanged")
st.write("The distribution of Rhodium shows a massively uneven distribution. We will asses it's outlier count in the "
         "next step, and if the outlier count is not favorable, we will remove 'RHO_PRICE'")
st.divider()


st.write("# Step 7: Outlier Analysis")
st.write("This step is to identify the number of outliers contained in each variable. We will use this in conjunction "
         "with the visualisation in the previous step to assess whether to remove or keep outliers present "
         "in the dataset")

Q1 = continuous_variables.quantile(0.25)
Q3 = continuous_variables.quantile(0.75)
IQR = Q3 - Q1

outliers = (continuous_variables < (Q1 - 1.5 * IQR)) | (continuous_variables > (Q3 + 1.5 * IQR))

outliers_count = outliers.sum()

st.write("#### Number of outliers for each continuous predictor variable")
dtype_df_outliers = pd.DataFrame(outliers_count, columns=["Number of Outliers"]).reset_index()
dtype_df_outliers = dtype_df_outliers.rename(columns={"index": "Column Name"})
st.dataframe(dtype_df_outliers, use_container_width=True)

st.write("### Observations of Step 7:")
st.write("After visualising the distribution in the previous step, and assessing how many outliers were present in the",
         "variables relating to gold price, we have elected to maintain the distribution as is, as it will not skew"
         " the predicted value too much in later analysis.")
st.write("As stated in the previous step, we have assessed the outlier count for RHO_PRICE, and given it is nearly "
         "as much as that of gold, without having the favourable distribution of gold, we have elected to remove it.")

continuous_variables = continuous_variables.drop(columns=["RHO_PRICE"])
cleaned_data = cleaned_data.drop(columns=["RHO_PRICE"])
st.divider()


st.write("# Step 8: Missing Value Analysis")

st.write("For this step, we will examine how many missing values are present in each variable, and if the number is low"
         " look to remove the rows containing the missing values. If there are many missing values, depending on the "
         "type of variable, we will look to overwrite the missing value with either the median value for continuous "
         "variables or the mode value for categorical variables.")

dtype_dt_missing = pd.DataFrame(cleaned_data.isnull().sum(), columns=["Missing Values"]).reset_index()
dtype_df_missing = dtype_dt_missing.rename(columns={"index": "Missing Values"})
st.dataframe(dtype_dt_missing, use_container_width=True)

st.write("### Observations of Step 8:")
st.write("For the sample data set being used for this model there are no missing values, the data set passed on to the "
         "machine learning models is unchanged")
st.divider()


st.write("# Step 9: Feature Selection")
st.write("For this step we will visualise the relationship between the continuous predictor variables using scatter "
         "plots, and the categorical variables using box plots.")

variables_to_visualise=[]
for variable in continuous_variables:
    if variable in variables_to_keep:
        variables_to_visualise.append(variable)

st.write("#### Visualising Continuous Predictor Variables VS Target Variable:")
st.write("In section, we will visualise the relationship between the continuous predictor variables and the target "
         "variable. What we are looking for here is whether there is a trend to the data. "
         "Depending on the type of trend, this can indicate one of three things:")
st.write("- Increasing Trend: If there is an increasing trend to the data, it means both variables are positively "
         "correlated. In layman's terns, they are directly proportional to each other. Where one increases, "
         "the other does as well")
st.write("- Decreasing Trend: If there is a decreasing trend to the data, it means both variables are negatively "
         "correlated. in layman's terms, where one increases, the other decreases. This is an outcome we might want"
         " to see, as it also helps ML models ")
st.write("- No Trend: If there is no trend to the data, it means there is no correlation between the predictor and "
         "target variables, and the predictor should be excluded from further analysis.")
st.write("Once we have visualised the relationship with all the predictor variables, we will assess which ones we need "
         "to investigate the correlation for further.")

for predictor in variables_to_visualise:
    fig, ax = plt.subplots(figsize=(20, 3))
    sns.scatterplot(cleaned_data, x=predictor, y='Adj Close')
    plt.title(f"{predictor} vs Adj Close")
    st.pyplot(fig)

st.write("Looking at the above scatter plots, we can see a clear trend with the variables relating to gold and "
         "platinum.")
st.write("The relationship between the target and palladium predictor variables however, don't clearly show any trend "
         "at all, so we will have a closer look at the correlation. At this step, we will reintroduce the other "
         "variables that were removed to investigate their Pearson's correlation coefficient to see whether "
         "we should include them in final analysis. To save space on the correlation heatmap, we will only include"
         " the other variables in the tables below.")

numeric_variables = cleaned_data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_variables.corr()

target_correlations = correlation_matrix['Adj Close'].drop('Adj Close')

st.write("##### Correlation of Continuous Predictor Variables VS Target Variable:")

strong_corr = target_correlations[target_correlations.abs() >= high]
moderate_corr = target_correlations[(target_correlations.abs() >= low) & (target_correlations.abs() < high)]
weak_corr = target_correlations[target_correlations.abs() < low]

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, ax=ax, cmap="rocket_r")
st.pyplot(fig)

numeric_variables = cleaned_data_copy.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_variables.corr()
target_correlations = correlation_matrix['Adj Close'].drop('Adj Close')

strong_corr_2 = target_correlations[target_correlations.abs() >= high]
moderate_corr_2 = target_correlations[(target_correlations.abs() >= low) & (target_correlations.abs() < high)]
weak_corr_2 = target_correlations[target_correlations.abs() < low]

st.write(f"###### Strong Correlations (|correlation| >= {high}):")
st.dataframe(strong_corr_2, use_container_width=True)

st.write(f"###### Moderate Correlations ({low} <=|correlation| < {high}):")
st.dataframe(moderate_corr_2, use_container_width=True)

st.write(f"###### Weak Correlations (|correlation| <{low}):")
st.dataframe(weak_corr_2, use_container_width=True)

# Creating a list out of the strong correlation variables to iterate over later
strong_cor_col_df = strong_corr_df.pivot_table(index='Var', columns='Var', values='Adj Close')
strong_cor_list = strong_cor_col_df.columns.tolist()

cleaned_data = cleaned_data.drop(columns=["PLD_Open", "PLD_High", "PLD_Low", "PLD_Price"])
continuous_variables = continuous_variables.drop(columns=["PLD_Open", "PLD_High", "PLD_Low", "PLD_Price"])

st.write("So that we can ensure we are conducting thorough investigation into potential MLM to use later, we will have"
         " a look at the distribution of the strong corr again, looking to see if there is a higher-degree polynomial"
         " than shown for the variables relating to gold. This will allow us to understand if we need to create a"
         " polynomial regression model. The results of this will be used in Step 13.")

for predictor in strong_cor_list:
    if numeric_variables[predictor].dtype == ('float64' or 'int64'):
        fig, ax = plt.subplots(figsize=(20, 12))
        sns.regplot(numeric_variables, x=predictor, y='Adj Close',order=2)
        plt.title(f"{predictor} vs Adj Close with estimated regression")
        st.pyplot(fig)

st.write("#### Visualising Categorical Predictor Variables VS Target Variable:")
st.write('In this section we will visualise the correlation of the target variable and the categorical variables'
         ' using box plots.')

for cat_col in bool_variables:
    fig, ax = plt.subplots(figsize=(20, 6))
    sns.boxplot(x=cat_col,y='Adj Close', data=cleaned_data)
    ax.set_title(f"{cat_col} vs Adj Close")
    st.pyplot(fig)

st.write("As the box plots don't show us much in the way of correlation between the predictors and the target variable,"
         " we will perform an ANOVA analysis to assess the correlation further")
st.write("In the ANOVA test, we are checking to see whether the mean values are significantly different between the"
         " target and the categorical variables. The indicator we are looking at in order to reject the null hypothesis"
         " is a low p-value.")

anova_results = []
for cat_col in bool_variables:
    anova_groups = cleaned_data.groupby(cat_col)['Adj Close'].apply(list)
    f_val, p_val = stats.f_oneway(*anova_groups)

    anova_results.append({"Categorical Variable": cat_col, "F-Value": f_val, "P-Value": p_val})

anova_df = pd.DataFrame(anova_results)

st.write("###### ANOVA Results")
st.write("The following table shows F and P-values for each categorical variable.")
st.dataframe(anova_df, use_container_width=True)

significant_vars = anova_df[anova_df["P-Value"] < 0.05]
st.write("### Significant Variables (P < 0.05):")
if not significant_vars.empty:
    st.dataframe(significant_vars, use_container_width=True)
else:
    st.write("No significant variables (P < 0.05) detected")

for var in bool_variables:
    if var not in significant_vars.columns:
        cleaned_data = cleaned_data.drop(columns=[var])

st.write("### Observations from Step 9:")
st.write("From the visualisation of the correlation between the continuous predictor variables and the target variable,"
         " we have selected only the variables with strong correlation (above 0.9 Pearson correlation index) to pass "
         "on to the machine learning models.")
st.write("Upon analysis of the correlation matrix, it appears that we chose the wrong variables to compare the price of"
         " gold to, as platinum falls within the mid range correlation. The variables with a high correlation value that"
         " we will use later are:", strong_cor_list)
st.write("Of the categorical predictor variables, while the box charts did not provide useful information regarding "
         "the correlation to the target variable, the ANOVA test results show that there is enough evidence to reject "
         "the null hypothesis in regard to some of the categorical variables. To this end, we will include "
         "'OF_Trend' and 'OS_Trend' in the final prediction")

st.divider()


st.write("# Step 10: Selecting final predictors for MLM")
st.write("In this step we will select the final predictor variables from the results of the past few steps. "
         "We will take the continuous variables that had the highest correlation value, and the categorical "
         "variables that had the lowest p-value.")

best_categorical = ['OF_Trend', 'OS_Trend']
final_predictor_variables = []
for item in best_categorical:
    final_predictor_variables.append(item)
for item in strong_cor_list:
    final_predictor_variables.append(item)

st.write("From our analysis, the best predictor variables are: ", final_predictor_variables)

# Adding the target variables into the final predictors to drop only columns not needed
final_predictor_variables.append("Adj Close")

for col in cleaned_data_copy.columns:
    if col not in final_predictor_variables:
        cleaned_data_copy = cleaned_data_copy.drop(columns=[col])

final_predictor_variables.remove("Adj Close")

st.divider()


st.write("# Step 11: Data Conversion to Numeric Values for ML Analysis")
st.write("For this step we will pass all the variables through a method to convert the datatypes to numeric one last"
         " time to ensure the MLM is able to properly use them to conduct its analysis")

for col in cleaned_data_copy.select_dtypes(include=['bool']).columns:
    cleaned_data_copy[col] = cleaned_data_copy[col].astype('int64')

predictor_dtype_df = pd.DataFrame(cleaned_data_copy.dtypes, columns=["data type"]).reset_index()
predictor_dtype_df = predictor_dtype_df.rename(columns={"index": "Column Name"})
st.dataframe(predictor_dtype_df, use_container_width=True)

st.write("As seen above, all the boolean values have been converted into integers")
st.divider()


st.write("# Step 12: Train/Test Data Split and Standardisation/Normalisation of Data")
st.write("For this step, we will select a portion of the dataset to use in training the machine "
         "learning models. As we are likely to use a linear regression model, this step may be unnecessary, "
         "however due to the size of the dataset, it will save on computing time to perform tests on a smaller dataset."
         , "As we are setting the threshold for how much data to test, we will use 1/3 of the dataset for ML training.")

test_size = (1/3)
x = cleaned_data_copy[final_predictor_variables]
y = cleaned_data_copy['Adj Close']

# We'll arbitrarily choose a random_state to ensure results are consistent upon repeat.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state=42)


scaler, x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
st.divider()


st.write("# Step 13: Selecting MLM to train on the data")
st.write('For this step, we will investigate which MLM are available to us. From the sklearn library, we '
         'have access to a few models. Given the results from Step 9, and viewing the estimated regression, '
         'there is no need for us to use a polynomial regression, so we will use the standard Linear Regression Model'
         ' (LRM) as our LRM')
st.write('To ensure we are selecting the best model, we wil also train 4 other models on the data and compare their '
         'results to determine the final model')
st.write('The selected models we have chosen are:')
st.write('- Linear Regression: A model most widely used when there is a linear relationship between '
         'target and predictor variables')
st.write('- Support Vector Regression (SVR): Typically used for classification tasks, however can also be used for '
         'regression models.')
st.write('- Decision Tree Regression: A type of regression model that builds a decision tree to predict the target'
         ' variable')
