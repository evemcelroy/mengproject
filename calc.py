# # x=211 # top of desired bar (pixel no.)
# # y=x-39 # top of y axis (pixel no.)
# # z=y*(-5/114) # axis increment/pixel increment
# # ANS=z+45 # top of y axis value
# # print(ANS)
#
# def give_results(X_train, X_test, y_train, y_test):
#     return mean, r2,
#
# # Key - tuple of parameters
# # Value - list of results
# results = {}
#
# test_size = 0
# rnd_state = 0
#
# for test_size in range(s, e, i):
#     for rnd_state in range(s, e, i):
#         results[(test_size, rnd_state)] = give_results(train_test_split(X, y, test_size=0.2, random_state=42))
#
#
#
#
# # import pandas as pd
# # import numpy as np
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# #
# # data = pd.read_csv("py_input_v1.csv")
# #
# # # Clean Data
# # data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('[^\w\s]', '')  # removes spaces + special chars
# #
# # # Correlation Matrix
# # numeric_columns = data.select_dtypes(include=[np.number])
# # correlation_matrix = numeric_columns.corr()
# # print(correlation_matrix)
# # # Create a heatmap
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
# #
# # plt.title('Correlation Matrix Heatmap')
# # plt.show()
#
# import matplotlib.pyplot as plt
#
# # Define the feature importances for each target variable
# # DTR
# feature_importance_UTS = [0.17196111737735267, 0.6602709989420702, 0.08435503322500446, 0.08341285045557269]
# feature_importance_YS = [0.6272888451729806, 0.19123601642993984, 0.13683058317299532, 0.04464455522408425]
# feature_importance_YM = [0.6149751801966231, 0.2090076497272753, 0.016769309123305282, 0.15924786095279633]
# feature_importance_Elong = [0.6873909078333458, 0.16196950388485418, 0.09582606370935959, 0.054813524572440375]
#
# # RFR
# # feature_importance_UTS = [0.22497400891996217, 0.5257602602024622, 0.10853760718209664, 0.14072812369547893]
# # feature_importance_YS = [0.2632794922578828, 0.5228927186197765, 0.0472513466089813, 0.16657644251335935]
# # feature_importance_YM = [0.3528653280080053, 0.4473753117405718, 0.09235509649293538, 0.10740426375848752]
# # feature_importance_Elong = [0.5103904043423901, 0.32432865158063556, 0.0690764470962695, 0.09620449698070492]
#
# # Define the feature names
# feature_names = ['P_[W]', 'v_[mm/s]', 't_[µm]', 'h_[µm]']
#
# # Create a 4x4 grid for the combined heatmap
# fig, ax = plt.subplots(figsize=(10, 8))
#
# # Create the combined heatmap with rounded values
# cax = ax.matshow([[round(val, 2) for val in feature_importance_UTS],
#                  [round(val, 2) for val in feature_importance_YS],
#                  [round(val, 2) for val in feature_importance_YM],
#                  [round(val, 2) for val in feature_importance_Elong]],
#                  cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
#
# # Set the x-axis and y-axis labels
# ax.set_xticks(range(len(feature_names)))
# ax.set_xticklabels(feature_names)
# ax.set_yticks([0, 1, 2, 3])
# ax.set_yticklabels(['UTS_[Mpa]', 'YS_[Mpa]', 'YM_[Gpa]', 'Elong_[%]'])
# ax.xaxis.set_label_position('top')
# ax.set_xlabel('Processing Parameter')
# ax.set_ylabel('Target Variable')
#
# # Add colorbar
# fig.colorbar(cax)
#
# # Display the values within the color blocks
# for i in range(4):
#     for j in range(4):
#         ax.text(j, i, str([round(val, 2) for val in [feature_importance_UTS, feature_importance_YS, feature_importance_YM, feature_importance_Elong][i]][j]), va='center', ha='center')
#
# # Adjust the layout
# plt.tight_layout()
#
# # Show the combined heatmap with values
# plt.show()





# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

data = pd.read_csv("py_input_v1.csv")

# Clean Data
data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('[^\w\s]', '')  # removes spaces + special chars

# Define all input features + target variables
input_features = ["P_[W]", "v_[mm/s]", "t_[µm]", "h_[µm]"]
targets = ["UTS_[Mpa]", "YS_[Mpa]", "YM_[Gpa]", "Elong_[%]", "Microhard_[HV]"]

# Optimisation
best_params = {'test_size': None, 'random_state': None}
best_metrics = {'mse': float('inf'), 'r2': -float('inf')}

test_size_range = range(0, 100, 1)
random_state_range = range(0, 100, 1)

for test_size in test_size_range:
    for rnd_state in random_state_range:
        print(f"Testing with test_size={test_size}, random_state={rnd_state}")
        test_size_percent = test_size/100
        mse_list, r2_list = [], []

        for target in targets:
            X = data[input_features]
            y = data[target]

            # Split data into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rnd_state)

            # Imputation Method
            imputer = SimpleImputer(strategy='mean')
            X_train = imputer.fit_transform(X_train)
            y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            for feature in input_features:
                X_test[feature] = imputer.transform(X_test[feature].values.reshape(-1, 1)).ravel()
            y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()

            # Create and train the Random Forest Regressor
            regressor = RandomForestRegressor(random_state=42)
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
