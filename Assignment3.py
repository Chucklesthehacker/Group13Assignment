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

# Features to add /things to check:
#  Remove duplicate columns
#  A high strength correlation matrix
#  A genralized way to filter out boolean data
#  Check the number of steps is correct
# remove the file selector, we know what file we want to use
# Maybe add a region selector to filter out open/high/low and close for the region by column name. Might need to create a static dictionary for it or smth

# correlation strength variables
high = 0.95
low =0.75


gold_data = pd.read_csv('FINAL_USO.csv',index_col='Date', infer_datetime_format=True)

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

# A function to clean the data by converting object columns to suitable data types
def clean_dataset(data):
    bool_variables = ['EU_Trend', 'OF_Trend','OS_Trend', 'SF_Trend', 'USB_Trend', 'PLT_Trend', 'PLD_Trend', 'USDI_Trend']
    for variable in bool_variables:
        data[variable] = data[variable].astype('bool')

    return data


dataset = gold_data
cleaned_data = dataset.drop_duplicates()

# Convert object data types to numeric or datetime
cleaned_data = clean_dataset(cleaned_data)
st.title(f'Processing Dataset: FINAL_USO')

st.write('### Sample Data from the Selected Dataset')
st.dataframe(cleaned_data.sample(5), use_container_width=True)

# Show cleaned data
st.write('### Cleaned data types')
dtype_df = pd.DataFrame(cleaned_data.dtypes, columns=["data type"]).reset_index().rename(
    columns={"index": "Column Name"})
st.dataframe(dtype_df, use_container_width=True)

drop_duplicate_data = cleaned_data.drop_duplicates()

# Show shape and columns of selected dataset
st.write('## Dataset Information')
st.write(f'### Shape: {cleaned_data.shape}')
st.write(f'### Shape after deleting duplicates: {drop_duplicate_data.shape}')
st.write(f'Columns in the dataset: ', cleaned_data.columns.tolist())

# Removing unnecessary columns
st.write("### Select Columns to remove from the dataset")
remove_columns = st.multiselect("Select columns to remove:", cleaned_data.columns)
count = 0
if remove_columns:
    for col in cleaned_data.columns:
        if col in remove_columns:
            cleaned_data = cleaned_data.drop(columns=[col])
            count = count+1
    st.write(f"Removed  {remove_columns} ")
    st.write(f" Shape after removing columns: {cleaned_data.shape}")



# Select the Target Variable
st.write('## Select the Target Variable')
target_variable = st.selectbox('Select the Target Variable:', options=list(cleaned_data.columns))
st.write(f"### Target Variable '{target_variable}'")

st.write("## Visualizing the Target Variable")

fig, ax = plt.subplots()
ax.hist(cleaned_data[target_variable], bins=50, edgecolor='black', alpha=0.5)
ax.set_title(f'Distribution of {target_variable}')
ax.set_xlabel(target_variable)
ax.set_ylabel('Frequency')
st.pyplot(fig)

# st.write("### Remove extra columns")
# keeping_columns = st.multiselect("Select columns to keep:", cleaned_data.columns)
# count = 0
# for col in cleaned_data.columns:
#     if col not in keeping_columns:
#         cleaned_data = cleaned_data.drop(columns=[col])
#         count = count+1
# st.write(f"### Removed extra {count} variables")
# st.write("Remaining Variables: ", cleaned_data.columns.tolist())
# #
# st.write('### Outlier Analysis: ')
# unique_values = []
# for col in cleaned_data.columns:
#     if cleaned_data.nunique() < 1000:
#         unique_values = unique_values.append(col)
#
# count =0
# for col in cleaned_data.columns:
#     if col not in unique_values:
#         cleaned_data = cleaned_data.drop(columns=[col])
#         count +=1
#
# st.write(f"### Removed {count} columns with less than 1000 unique values")


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
    st.write("### Unique Values")
    st.dataframe(cleaned_data.nunique(), use_container_width=True)

st.write("### Summary Statistics:")
st.dataframe(cleaned_data.describe(), use_container_width=True)

# Creating a buffer to pipe the output of info() method to show in streamlit
buffer = io.StringIO()
cleaned_data.info(buf=buffer)
cleaned_data_info = buffer.getvalue()
st.write("### Description of Dataset")
st.text(cleaned_data_info)


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
st.write("### Select Columns to keep")
keeping_columns = st.multiselect("Select columns to keep:", cleaned_data.columns)
count = 0
if keeping_columns:
    for col in cleaned_data.columns:
        if col not in keeping_columns:
            cleaned_data = cleaned_data.drop(columns=[col])
            count = count+1
    st.write(f"### Removed extra {count} variables")
    st.write("Remaining Variables: ", cleaned_data.columns.tolist())

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

# WE NEED TO COME BACK AND LOOKY AT THISSY IT WORKS BUT EHEHEHEHEH
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
        else:
            st.write("No categorical variables selected for ANOVA")
    else:
        st.write("The target variable must be continuous for ANOVA analysis")
else:
    st.write("No categorical available for ANOVA")

# Step 10:
st.write("## Step 10: Selecting Final Predictors")
for columns in cleaned_data:
    if columns in target_variable:
        final_predictors = cleaned_data.drop(columns=[columns])


# Ensure that numeric columns are selected
selected_features = st.multiselect("Select predictor variables (independent variables):",
                                   final_predictors.select_dtypes(include=['float64', 'int64']).columns)
# exclude target variable
st.write("### Selected Features: ", selected_features)


if selected_features:
    st.write(f"### Target Variable: '{target_variable}'")

    # step 11: data prep for machine learning
    st.write('## Step11: Data Preparation for Machine Learning')

    X = cleaned_data[selected_features]
    y = cleaned_data[target_variable]

    test_size = st.slider("Select the test size (percentage):",min_value=0.1, max_value=0.5, value=0.2, step=0.01)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
# NOT ACTUALLY STEP 12
    st.write("## Step (12): Model Training and Evaluation")

    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = train_models(X_train_scaled, Y_train)

    trained_models = st.session_state.trained_models

    model_performance = {}
    for name, model in trained_models.items():
        model.fit(X_train_scaled, Y_train)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        mae = mean_absolute_error(Y_test, y_pred)

        model_performance[name] = {"MSE": mse, "R2 Score": r2, "MAE": mae}

    performance_df = pd.DataFrame(model_performance).T

    styled_df = performance_df.style.format(precision=5)\
        .background_gradient(subset=['MSE'], cmap="Purples", low=0, high=1) \
        .background_gradient(subset=['R2 Score'], cmap="Greens", low=0, high=1) \
        .background_gradient(subset=['MAE'], cmap="Oranges", low=0, high=1) \
        .set_properties(**{'text-align': 'center'}) \
        .set_table_styles([{
        'selector': 'th',
        'props': [('font-size', '14px'), ('text-align', 'center'), ('color', '#ffffff'),
                  ('background-color', '#404040')]
    }])

    # Display the table with st.dataframe
    st.write("## Model Performance Table")
    st.dataframe(styled_df, use_container_width=True)

    st.write("## Visualizing Model Performance Comparison")

    # Extracting model names and their respective performance metrics
    model_names = list(model_performance.keys())
    mse_values = [model_performance[model]["MSE"] for model in model_names]
    r2_values = [model_performance[model]["R2 Score"] for model in model_names]
    mae_values = [model_performance[model]["MAE"] for model in model_names]

    # Creating a bar plot to compare MSE, R2, and MAE across models
    fig, ax = plt.subplots(3,1, figsize=(14, 10))

    # MSE Comparison
    ax[0].bar(model_names, mse_values, color="purple")
    ax[0].set_title('Model Comparison: MSE (Mean Squared Error)')
    ax[0].set_ylabel('MSE')

    # R2 Score Comparison
    ax[1].bar(model_names, r2_values, color="green")
    ax[1].set_title('Model Comparison: R2 Score')
    ax[1].set_ylabel('R2 Score')

    ax[2].bar(model_names, mae_values, color="orange")
    ax[2].set_title('Model Comparison: MAE (Mean Absolute Error)')
    ax[2].set_ylabel('MAE')

    #display
    plt.tight_layout()
    st.pyplot(fig)

    st.write("## Selecting the Best Model")

    if model_performance:
        best_model_mse = min(model_performance, key=lambda x: model_performance[x]["MSE"])
        st.write("## Best Model based on lowest MSE: ", best_model_mse)

        st.write('## Retraining the Best Model')
        best_model = trained_models[best_model_mse]

        X_combined_scaled = scaler.fit_transform(X)
        best_model.fit(X_combined_scaled, y)

        # save the best model in session state and also as a file
        if 'best_model' not in st.session_state:
            st.session_state.best_model = best_model

        # save the model after retraining
        model_filename = 'best_model.pkl'
        joblib.dump(best_model, model_filename)
        st.write(f"Model '{best_model_mse}' has been retrained and saved as '{model_filename}'")

    else:
        st.write("No model performance results available. Please ensure (super)models were trained successfully")

    # Step ???
    st.write("## Step 15: Model Deployment - Predict Using Saved Model")

    # Load saved model
    model_filename = 'best_model.pkl'
    try:
        loaded_model = joblib.load(model_filename)
        st.write(f"Model '{model_filename}' has been loaded successfully")

        # Allow input of values for features
        st.write("### Provide the input values for prediction")

        # Generate input fields dynamically based on selected features
        user_input_vales = {}
        for feature in selected_features:
            user_input_vales[feature] = st.number_input(f"enter value for {feature}",
                                                        value=float(cleaned_data[feature].mean()))
        if st.button("Predict"):
            # Convert user input to dataframe
            user_input_df = pd.DataFrame([user_input_vales])

            # Scale the user inputs using the same scaler
            user_input_scaled = scaler.transform(user_input_df)

            # Make predictions using the loaded model
            predicted_value = loaded_model.predict(user_input_scaled)

            # Display the predicted value
            st.write(f"# Predicted final {target_variable}: ${predicted_value[0]:.2f}")

            # fig,ax = plt.subplots()
            #
            # feature_names = list(user_input_vales.keys())
            # feature_values = list(user_input_vales.values())
            #
            # ax.bar(feature_names, feature_values, color="lavender", label='Feature Values')
            #
            # ax.bar(['Predicted ' + target_variable],[predicted_value[0]], color='teal',
            #        label='Predicted Values')
            #
            # ax.set_xlabel('Value')
            # ax.set_title(f"Input Features and Predicted {target_variable}")
            # ax.legend()
            #
            # st.pyplot(fig)
            #
            # fig,ax = plt.subplots(figsize=(14, 10))
            #
            # copy_feature_name = feature_names.copy()
            # copy_feature_values = feature_values.copy()
            # # Add predicted value at the end
            # copy_feature_name.append(f"Predicted {target_variable}")
            # copy_feature_values.append(predicted_value[0])
            #
            # ax.plot(copy_feature_name, copy_feature_values, color='purple', marker='o', linestyle='-',
            #         label='Features Predicted Values')
            #
            # ax.set_xlabel('Features and Predicted value')
            # ax.set_ylabel('Values')
            # ax.set_title(f"Input Features and Predicted {target_variable}")
            # ax.grid(True)
            #
            # for i,txt in enumerate(copy_feature_values):
            #     ax.annotate(f"{txt:.2f}", (copy_feature_name[i], copy_feature_values[i]),
            #                 textcoords='offset points',
            #                 xytext=(0,10), ha='center', )
            #
            # st.pyplot(fig)

    except FileNotFoundError:
        st.write(f"Model '{model_filename}' does not exist. Please ensure 'twas save correctly milord")

else:
    st.write("No numeric features selected for training")
