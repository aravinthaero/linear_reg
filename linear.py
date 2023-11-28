import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title and introductory text
st.title("Linear Regression App")
st.markdown("Hi there! In this app, you can perform linear regression on your data.")

# Upload a CSV file
st.sidebar.markdown("### Upload Data")
data = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])


# Check if a file has been uploaded
if data:
    st.sidebar.markdown("### Data Overview")
    df = pd.read_csv(data)
    st.write(df.head())

    # Data exploration
    st.sidebar.subheader("Data Exploration")
    if st.sidebar.checkbox("Show Data Info"):
        st.write(df.info())

    if st.sidebar.checkbox("Show Data Summary"):
        st.write(df.describe())

    # Select X and y
    st.sidebar.subheader("Choose Features and Target")
    feature = st.sidebar.selectbox("Select a feature (X):", df.columns)
    target = st.sidebar.selectbox("Select the target variable (y):", df.columns)

    # Data visualization
    st.sidebar.subheader("Data Visualization")
    if st.sidebar.checkbox("Pairplot"):
        sns.pairplot(df)
        st.pyplot()

    # Train the linear regression model
    st.sidebar.subheader("Train the Model")
    if st.sidebar.checkbox("Train Linear Regression Model"):
        X = df[feature].values.reshape(-1, 1)
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = LinearRegression()
        model.fit(X_train, y_train)

        st.success("Model trained successfully!")

    # Predictions and visualization
    st.sidebar.subheader("Make Predictions")
    if st.sidebar.checkbox("Predictions"):
        y_pred = model.predict(X_test)

        st.write("Actual vs. Predicted values:")
        predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.write(predictions_df)

        # Plot the regression line
        st.write("Regression Line:")
        plt.scatter(X_test, y_test, color="gray")
        plt.plot(X_test, y_pred, color="red", linewidth=2)
        plt.xlabel(feature)
        plt.ylabel(target)
        st.pyplot()

# Custom code snippet
if st.sidebar.checkbox("Show Code Snippet"):
    st.code(
        """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your dataset (e.g., df)
# Define X and y
X = df[feature].values.reshape(-1, 1)
y = df[target].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
        """
    )
