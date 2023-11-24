import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv("py_input_v1.csv")

# Clean column names by removing spaces and special characters
data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('[^\w\s]', '')

# Define all input features and the target variables
input_features = ["P_[W]", "v_[mm/s]", "t_[µm]", "h_[µm]"]
targets = ["UTS_[Mpa]", "YS_[Mpa]", "YM_[Gpa]", "Elong_[%]"]

# Loop through the target variables
for target in targets:
    print(f"Model for target variable: {target}")

    X = data[input_features]
    y = data[target]

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values with mean value
    imputer = SimpleImputer(strategy='mean')
    # Fit and transform the imputer on the training set
    X_train = imputer.fit_transform(X_train)
    y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    # Transform the test set using the same imputer
    for feature in input_features:
        X_test[feature] = imputer.transform(X_test[feature].values.reshape(-1, 1)).ravel()
    y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()

    # Create and train the Random Forest Regressor
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    # Get feature importances
    feature_importances = regressor.feature_importances_

    # Pair feature names with their importances and print
    feature_importance_dict = dict(zip(input_features, feature_importances))
    print("Feature Importances:")
    for feature, importance in feature_importance_dict.items():
        print(f"{feature}: {importance}")

    # Plot the predicted values against actual values
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual for {target}")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.show()