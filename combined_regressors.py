# CTRL+/ COMMENT OUT
# SHIFT f10 RUN

# Import  Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
import seaborn as sns
import builtins

# Open a file for writing results to
with open("test_results.txt", "w") as f:

    # Redirect print to the file
    def print_to_file(*args, **kwargs):
        with open("test_results.txt", "a") as file:
            builtins.print(*args, **kwargs, file=file)


    print = print_to_file


data = pd.read_csv("py_input_v1.csv")
data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('[^\w\s]', '') # removes spaces + special chars

# Summary Statistics
pd.set_option('display.max_columns', None)
print(data.describe())

# Correlation Matrix
numeric_columns = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Define all input features + target variables
input_features = ["P_[W]", "v_[mm/s]", "t_[µm]", "h_[µm]"]
targets = ["UTS_[Mpa]", "YS_[Mpa]", "YM_[Gpa]", "Elong_[%]", "Microhard_[HV]"]

# Optimisation
best_params = {'test_size': None, 'random_state': None}
best_metrics = {'mse': float('inf'), 'r2': -float('inf')}
test_size_range = np.arange(0.01, 1.0, 0.01)
random_state_range = range(0, 100, 1)
for test_size in test_size_range:
    for rnd_state in random_state_range:
        print(f"Testing with test_size={test_size}, random_state={rnd_state}")
        test_size_percent = test_size/100
        mse_list, r2_list = [], []

        for target in targets:
            X = data[input_features]
            y = data[target]

# # Plot Predicted v Actual
# fig, axes = plt.subplots(3, 2, figsize=(12, 10))
# fig.suptitle("Model predictions", fontsize=18)
#
# # Loop through target variables
# for i, target in enumerate(targets):
#     print(f"Model for target variable: {target}")
#
#     X = data[input_features]
#     y = data[target]

    # Split data into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rnd_state)

            # Imputation Method
            imputer = SimpleImputer(strategy='mean')
            # Fit and transform the imputer on the training set
            X_train = imputer.fit_transform(X_train)
            y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            # Transform the test set using the same imputer
            for feature in input_features:
                X_test[feature] = imputer.transform(X_test[feature].values.reshape(-1, 1)).ravel()
            y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()

    # CHOOSE ML REGRESSION MODEL
            regressor = RandomForestRegressor(random_state=42)
            # regressor = DecisionTreeRegressor(random_state=42)
            regressor.fit(X_train, y_train)

    # Predict on the test set
            y_pred = regressor.predict(X_test)

    # Evaluation Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_list.append(mse)
            r2_list.append(r2)

        # Calculate mean metrics across targets
        avg_mse = np.mean(mse_list)
        avg_r2 = np.mean(r2_list)

        # Update best parameters if current metrics are better
        if avg_mse < best_metrics['mse'] and avg_r2 > best_metrics['r2']:
            best_params['test_size'] = test_size
            best_params['random_state'] = rnd_state
            best_metrics['mse'] = avg_mse
            best_metrics['r2'] = avg_r2

    # Print the best parameters and metrics
print("Best Parameters:")
print(f"test_size={best_params['test_size']}, random_state={best_params['random_state']}")
print("Best Metrics:")
print(f"Mean Squared Error: {best_metrics['mse']}")
print(f"R-squared: {best_metrics['r2']}")
del print

    # print("Mean Squared Error:", mse)
    # print("R-squared:", r2)

    # results[(test_size, rnd_state)] = (mse, r2)
    # for params, metrics in results.items():
    #     print(
    #         f"Parameters: test_size={params[0]}, random_state={params[1]}, Mean Squared Error: {metrics[0]}, R-squared: {metrics[1]}")

    # Feature Importances
feature_importances = regressor.feature_importances_
all_feature_importances = []

    # Pair feature names with their importances and print
feature_importance_dict = dict(zip(input_features, feature_importances))
print("Feature Importances:")
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")
all_feature_importances.append(feature_importances)

# Plot
row = i // 2
col = i % 2

ax = axes[row, col]
print(f"row {row} col {col} target {target}")

# Scatter plot
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title(f"Predicted vs Actual for {target}")

# Dashed line representing R-squared
r2 = r2_score(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label=f'R-squared: {r2:.2f}')
ax.legend()

plt.tight_layout()
plt.show()

########################################################################################################################
# Plot Feature Importances
fig, ax = plt.subplots(figsize=(12, 8))

# Normalize feature importances for better visualization
normalized_feature_importances = np.array(all_feature_importances) / np.sum(all_feature_importances, axis=1)[:, np.newaxis]

# Create the combined heatmap with rounded values
cax = ax.matshow(normalized_feature_importances, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

# Set the x-axis and y-axis labels
ax.set_xticks(range(len(input_features)))
ax.set_xticklabels(input_features)
ax.set_yticks(range(len(targets)))
ax.set_yticklabels(targets)
ax.xaxis.set_label_position('top')
ax.set_xlabel('Processing Parameter')
ax.set_ylabel('Target Variable')

# Display the values within the color blocks
for i in range(len(targets)):
    for j in range(len(input_features)):
        ax.text(j, i, f"{normalized_feature_importances[i][j]:.2f}", va='center', ha='center', color='black')
fig.colorbar(cax)
plt.tight_layout()
plt.show()
