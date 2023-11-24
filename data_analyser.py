# CTRL+/ COMMENT OUT
# SHIFT f10 RUN

# Import Key Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

data = pd.read_csv("py_input_v1.csv")

# Clean column names by removing spaces and special characters
data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('[^\w\s]', '')

# Explore Data
print(data.head())  # Display first few rows
pd.set_option('display.max_columns', None)
print(data.describe())  # Summary statistics - avg values

# Define the combinations for scatter plots
combinations = [
    (1, 6, 'YS v P', 1),
    (2, 6, 'YS v v', 2),
    (3, 6, 'YS v t', 3),
    (2, 7, 'YM v v', 4),
]

# Create a figure and axis
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
    for _, row_data in data.iterrows():
        ax[row, col].annotate(row_data.iloc[0], (row_data.iloc[x_col], row_data.iloc[y_col]))

plt.show()

# Using pandas for correlation coeffs between variables
numeric_columns = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)

####################################################################
# Define features we want to use in DecisionTree
features = ["P_[W]"]
target = "YS_[Mpa]"

X = data[features]
y = data[target]

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a SimpleImputer to impute missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on the training set
y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Transform the test set using the same imputer
y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()

# Create and train the Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
#######################################################################################
