# CTRL+/ COMMENT OUT
# SHIFT f10 RUN

# Import Key Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("py_input_v1.csv")

# Clean column names by removing spaces and special characters
data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('[^\w\s]', '')

# Summary Statistics
pd.set_option('display.max_columns', None)
print(data.describe())

# Define combinations of input and output variables
scatter_combinations = [("P_[W]", "YS_[Mpa]", "YS v P"),
                        ("v_[mm/s]", "YS_[Mpa]", "YS v v"),
                        ("t_[µm]", "YS_[Mpa]", "YS v t"),
                        ("v_[mm/s]", "YM_[Gpa]", "YM v v")]
# Loop through the combinations and create scatter plots
for x_col, y_col, title in scatter_combinations:
    plt.scatter(data[x_col], data[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Scatter Plot: {title}")
    plt.show()

#################################################################################################
# Random Forest Regressor

# Define all input features and the target variable
input_features = ["P_[W]", "v_[mm/s]", "t_[µm]", "h_[µm]"]
target = "YS_[Mpa]"

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