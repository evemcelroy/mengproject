# SHORTCUTS
# CTRL +/ comment out
# CTRL 0/ uncomment
# SHIFT F10 run

# Import  Libraries
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import MultiTaskLasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
import seaborn as sns
import builtins
import datetime

# Optimisation
# Redirect print to the file
# def print_to_file(*args, **kwargs):
#     with open("optimisation.txt", "a") as file:
#         builtins.print(*args, **kwargs, file=file)


def print_to_file(*args, **kwargs):
    with open("output.txt", "a") as file:
        builtins.print(*args, **kwargs, file=file)


print_to_file("ML to Determine Mechanical Properties of AMM")
print_to_file(f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_to_file("Author: Eve McElroy\n")

data = pd.read_csv("py_input_v1.csv")
data.columns = (
    data.columns.str.strip().str.replace(" ", "_").str.replace("[^\w\s]", "")
)  # removes spaces + special chars
# Calculate energy density and add it to the DataFrame
data["E_[J/mm^3]"] = data["P_[W]"] / (
    data["v_[mm/s]"] * data["t_[µm]"] * data["h_[µm]"]
)

# Specify powder grade for analysis
# filtered_data = data[data["Grade"] == 5].copy()  # Grade 5 only
# filtered_data = data[data["Grade"] == 23].copy()  # Grade 23 only
filtered_data = data  # Both grades

# filtered_data = data[data["Machine"] == "Concept"].copy()
# filtered_data = data[data["Machine"] == "EOS"].copy()


# Summary Statistics
pd.set_option("display.max_columns", None)
print(data.describe())
print_to_file("Summary Statistics:")
print_to_file(data.describe())
print_to_file("\n" + "=" * 50 + "\n")

# Correlation Matrix
numeric_columns = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix Heatmap")
plt.show()
print_to_file("Correlation Matrix:")
print_to_file(correlation_matrix)
print_to_file("\n" + "=" * 50 + "\n")

# PSD Curves
percentiles = [10, 50, 90]
percentile_columns = [f"D{p}_[µm]" for p in percentiles]
plt.figure(figsize=(10, 6))
x_values = np.linspace(
    filtered_data[percentile_columns].min().min(),
    filtered_data[percentile_columns].max().max(),
    1000,
)
for col in percentile_columns:
    curve_color = (
        "#FFAE90" if "D10" in col else ("#EF39A7" if "D50" in col else "#2494CC")
    )
    mean, std = filtered_data[col].mean(), filtered_data[col].std()
    plt.plot(
        x_values,
        norm.pdf(x_values, mean, std),
        color=curve_color,
        linewidth=2,
        label=f"{col}",
    )
plt.title("Particle Size Distribution Curves")
plt.xlabel("Particle Size (µm)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# D10 D50 D90
bins_d10 = [5, 10, 15, 20, 25]
bins_d50 = [10, 20, 30, 40]
bins_d90 = [20, 30, 40, 50, 60]
fig, axes = plt.subplots(3, 1, figsize=(7, 15))
for i, col in enumerate(percentile_columns):
    bar_color = (
        "#FFAE90" if "D10" in col else ("#EF39A7" if "D50" in col else "#2494CC")
    )
    n, bins, patches = axes[i].hist(
        filtered_data[col],
        bins=bins_d10 if "D10" in col else (bins_d50 if "D50" in col else bins_d90),
        align="left",
        rwidth=0.4,
        alpha=0.9,
        color=bar_color,
    )
    axes[i].set_xticks(bins[:-1] + 0.2)
    axes[i].set_xticklabels(
        [f"{int(value)}-{int(value + bins[1])}" for value in bins[:-1]]
    )
    average_value = filtered_data[col].mean()
    axes[i].axhline(
        average_value,
        color=bar_color,
        linestyle="--",
        linewidth=2,
        label=f"Average {col}: {average_value:.2f}",
    )
    axes[i].set_title(f"PSD - {col}")
    axes[i].set_xlabel("Particle Size (µm)")
    axes[i].set_ylabel("Frequency (count)")
    axes[i].grid(False)
    axes[i].legend(loc="upper left")
plt.tight_layout()
plt.show()

# Combined D10_50_90
bins_combined = list(range(0, 101, 10))  # Bins in increments of 10 microns
bar_width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
for i, col in enumerate(percentile_columns):
    bar_color = (
        "#FFAE90" if "D10" in col else ("#EF39A7" if "D50" in col else "#2494CC")
    )
    n, bins, patches = ax.hist(
        filtered_data[col],
        bins=bins_combined,
        align="left",
        rwidth=0.8,
        color=bar_color,
        label=f"{col}",
    )
    average_value = filtered_data[col].mean()
    ax.axhline(
        average_value,
        color=bar_color,
        linestyle="--",
        linewidth=2,
        label=f"Average {col}: {average_value:.2f}",
    )
ax.set_xticks(bins_combined[:-1] + [bins_combined[-1]])
ax.set_xticklabels(
    [f"{value}-{value + 10}" for value in bins_combined[:-1]] + [f"{bins_combined[-1]}"]
)
ax.set_title("Particle Size Distribution")
ax.set_xlabel("Particle Size (µm)")
ax.set_ylabel("Frequency (count)")
ax.legend()
ax.grid(False)
plt.tight_layout()
plt.show()

# Define all input features + target variables
input_features = ["P_[W]", "v_[mm/s]", "t_[µm]", "h_[µm]", "E_[J/mm^3]"]
targets = ["UTS_[Mpa]", "YS_[Mpa]", "YM_[Gpa]", "Elong_[%]", "Microhard_[HV]"]

# Optimisation
# best_params = {"test_size": None, "random_state": None}
# best_metrics = {"mse": float("inf"), "r2": -float("inf")}
# test_size_range = np.arange(0.025, 0.8, 0.025)
# rnd_state = 42
# for test_size in test_size_range:
#     print(f"Testing with test_size={test_size}, random_state={rnd_state}")
#     mse_list, r2_list = [], []
# for target in targets:
#     X = data[input_features]
#     y = data[target]

# Plot Predicted v Actual
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("Model predictions", fontsize=18)

all_feature_importances = []
# Loop through target variables
for i, target in enumerate(targets):
    print(f"Model for target variable: {target}")

    X = filtered_data[input_features]
    y = filtered_data[target]

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.125, random_state=42
    )

    # Imputation Method
    imputer = SimpleImputer(strategy="mean")
    # Fit and transform the imputer on the training set
    X_train = imputer.fit_transform(X_train)
    y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    # Transform the test set using the same imputer
    for feature in input_features:
        X_test[feature] = imputer.transform(
            X_test[feature].values.reshape(-1, 1)
        ).ravel()
    y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()

    # Choose ML Regression Model
    # regressor = DecisionTreeRegressor(random_state=42)
    regressor = RandomForestRegressor(random_state=42)
    # regressor = GradientBoostingRegressor(random_state=42)
    # regressor = KNeighborsRegressor(n_neighbors=5)

    regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = regressor.predict(X_test)

    # Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print_to_file(f"\nEvaluation Metrics for {target}:")
    print_to_file(f"R-squared: {r2:.2f}")
    print_to_file(f"MSE: {mse:.2f}")
    print_to_file("\n" + "=" * 50 + "\n")

    #     mse_list.append(mse)
    #     r2_list.append(r2)
    #
    # # Calculate mean metrics across targets
    # avg_mse = np.mean(mse_list)
    # avg_r2 = np.mean(r2_list)
    # Optimisation
    #     print_to_file(f"Test size: {test_size:.2f} yields mse {avg_mse} and r2 {avg_r2}")
    #     # Update best parameters if current metrics are better
    #     if avg_mse < best_metrics["mse"] and avg_r2 > best_metrics["r2"]:
    #         best_params["test_size"] = test_size
    #         best_metrics["mse"] = avg_mse
    #         best_metrics["r2"] = avg_r2
    # # Print the best parameters and metrics
    # print("Best Parameters:")
    # print(f"test_size={best_params['test_size']}")
    # print("Best Metrics:")
    # print(f"Mean Squared Error: {best_metrics['mse']}")
    # print(f"R-squared: {best_metrics['r2']}")
    # print("Mean Squared Error:", mse)
    # print("R-squared:", r2)
    # results[(test_size, rnd_state)] = (mse, r2)
    # for params, metrics in results.items():
    #     print(
    #         f"Parameters: test_size={params[0]}, random_state={params[1]}, Mean Squared Error: {metrics[0]}, R-squared: {metrics[1]}")

    # Feature Importances
    feature_importances = regressor.feature_importances_
    all_feature_importances.append(feature_importances)

    # Pair feature names with their importances and print
    feature_importance_dict = dict(zip(input_features, feature_importances))
    print("Feature Importances:")
    print_to_file(f"Feature Importances for {target}:")
    for feature, importance in feature_importance_dict.items():
        print(f"{feature}: {importance}")
    print_to_file("\n" + "=" * 50 + "\n")

    # Plot
    row = i // 2
    col = i % 2

    ax = axes[row, col]
    print(f"row {row} col {col} target {target}")

    # Predicted vs Actual
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Predicted vs Actual for {target}")

    # Dashed line representing R2
    r2 = r2_score(y_test, y_pred)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "k--",
        lw=4,
        label=f"R-squared: {r2:.2f}",
    )
    ax.legend()
plt.tight_layout()
plt.show()

# Plot Feature Importances
fig, ax = plt.subplots(figsize=(12, 8))
# Normalize feature importances for better visualization
normalized_feature_importances = (
    np.array(all_feature_importances)
    / np.sum(all_feature_importances, axis=1)[:, np.newaxis]
)
cax = ax.matshow(
    normalized_feature_importances, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1
)
ax.set_xticks(range(len(input_features)))
ax.set_xticklabels(input_features)
ax.set_yticks(range(len(targets)))
ax.set_yticklabels(targets)
ax.xaxis.set_label_position("top")
ax.set_xlabel("Processing Parameter")
ax.set_ylabel("Target Variable")

# Display the values within the color blocks
print(all_feature_importances)
for i in range(len(targets)):
    for j in range(len(input_features)):
        ax.text(
            j,
            i,
            f"{normalized_feature_importances[i][j]:.2f}",
            va="center",
            ha="center",
            color="black",
        )
fig.colorbar(cax)
plt.tight_layout()
plt.show()
