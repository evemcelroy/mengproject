# CTRL+/ COMMENT OUT
# SHIFT f10 RUN

# Import  Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

data = pd.read_csv("py_input_v1.csv")

# Clean Data
data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('[^\w\s]', '') # removes spaces + special chars

# Summary Statistics
pd.set_option('display.max_columns', None)
# print(data.describe())

# Correlation Matrix
numeric_columns = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)

# Scatter Plot Combinations
combinations = [(1, 6, 'YS v P', 1),
                (2, 6, 'YS v v', 2),
                (3, 6, 'YS v t', 3),
                (2, 7, 'YM v v', 4),]

fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Loop through combinations and create scatter plots
for i, (x_col, y_col, title, index) in enumerate(combinations):
    row = i // 2
    col = i % 2

    ax[row, col].scatter(data.iloc[:, x_col], data.iloc[:, y_col])
    ax[row, col].set_xlabel(data.columns[x_col])
    ax[row, col].set_ylabel(data.columns[y_col])
    ax[row, col].set_title(f'Scatter Plot: {title}')

    # Annotate the data point names
    # for _, row_data in data.iterrows():
    #     ax[row, col].annotate(row_data.iloc[0], (row_data.iloc[x_col], row_data.iloc[y_col]))

plt.show()

# Define all input features + target variables
input_features = ["P_[W]", "v_[mm/s]", "t_[µm]", "h_[µm]"]
targets = ["UTS_[Mpa]", "YS_[Mpa]", "YM_[Gpa]", "Elong_[%]"]

# Loop through target variables
for target in targets:
    print(f"Model for target variable: {target}")

    X = data[input_features]
    y = data[target]

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Imputation Method
    imputer = SimpleImputer(strategy='mean')
    # Fit and transform the imputer on the training set
    X_train = imputer.fit_transform(X_train)
    y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    # Transform the test set using the same imputer
    for feature in input_features:
        X_test[feature] = imputer.transform(X_test[feature].values.reshape(-1, 1)).ravel()
    y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()

    # Create and train the Decision Tree Regressor
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = regressor.predict(X_test)

    # Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    # Feature Importances
    feature_importances = regressor.feature_importances_

    # Pair feature names with their importances and print
    feature_importance_dict = dict(zip(input_features, feature_importances))
    print("Feature Importances:")
    for feature, importance in feature_importance_dict.items():
        print(f"{feature}: {importance}")

    # Plot Predicted v Actual
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual for {target}")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.show()


