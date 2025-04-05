
# Import pandas for data manipulation and analysis. 
import pandas as pd

# Import pandas for data manipulation and analysis.
import numpy as np

# Import matplotlib for creating static visualizations
import matplotlib.pyplot as plt

# Import seaborn for statistical data visualization (built on matplotlib)
import seaborn as sns

# Import the California housing dataset from sklearn
from sklearn.datasets import fetch_california_housing

# Import train_test_split for splitting data into training and test sets
from sklearn.model_selection import train_test_split

# Import LinearRegression for building a linear regression model
from sklearn.linear_model import LinearRegression

# Import performance metrics for model evaluation
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Import Pandas Plotting and Scatter Matrix
from pandas.plotting import scatter_matrix

# Import Stratified Shuffle Split
from sklearn.model_selection import StratifiedShuffleSplit

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# We Load the 'titantic' dataset via sns.load_dataset
titanic = sns.load_dataset('titanic')

#We retrieve its summary info via '.info()'
titanic.info()

# Here we 'print' the first 10 rows via '.head(10)'
print(titanic.head(10))

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Impute missing 'Age' values using median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# 2. Drop rows with missing 'fare' (or impute if preferred)
titanic.dropna(subset=['fare'], inplace=True)

# Alternatively, you can impute fare instead of dropping:
titanic['fare'].fillna(titanic['fare'].median(), inplace=True)

# 3. Create 'family_size' feature
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

# 4. Convert categorical features to numeric

# Convert 'sex' to binary
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

# One-hot encode 'embarked' (drop_first avoids multicollinearity)
titanic = pd.get_dummies(titanic, columns=['embarked'], drop_first=True)

# Preview the cleaned data
print(titanic.head())

# Case 1. age
X1 = titanic[['age']]
y1 = titanic['fare']
# Case 2. family_size
X2 = titanic[['family_size']]
y2 = titanic['fare']
# Case 3. age, family_size
X3 = titanic[['age', 'family_size']]
y3 = titanic['fare']
# Case 4. parch
X4 = titanic[['parch']]
y4 = titanic['fare']

# Train Case 1
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=123)
# Train Case 2
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=123)
# Train Case 3
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=123)
# Train Case 4
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=123)

# Linear Regression for Case 1
lr_model1 = LinearRegression().fit(X1_train, y1_train)
# Linear Regression for Case 2
lr_model2 = LinearRegression().fit(X2_train, y2_train)
# Linear Regression for Case 3
lr_model3 = LinearRegression().fit(X3_train, y3_train)
# Linear Regression for Case 4
lr_model4 = LinearRegression().fit(X4_train, y4_train)

# Predictions
# Case 1
y_pred_train1 = lr_model1.predict(X1_train)
y_pred_test1 = lr_model1.predict(X1_test)
# Case 2
y_pred_train2 = lr_model2.predict(X2_train)
y_pred_test2 = lr_model2.predict(X2_test)
# Predictions for Case 3
y_pred_train3 = lr_model3.predict(X3_train)
y_pred_test3 = lr_model3.predict(X3_test)
# Predictions for Case 4
y_pred_train4 = lr_model4.predict(X4_train)
y_pred_test4 = lr_model4.predict(X4_test)

# Evaluation for Case 1
print("Case 1: Training R²:", r2_score(y1_train, y_pred_train1))
print("Case 1: Test R²:", r2_score(y1_test, y_pred_test1))
print("Case 1: Test RMSE:", mean_squared_error(y1_test, y_pred_test1) ** 0.5)
print("Case 1: Test MAE:", mean_absolute_error(y1_test, y_pred_test1))
# Evaluation for Case 2
print("\nCase 2: Training R²:", r2_score(y2_train, y_pred_train2))
print("Case 2: Test R²:", r2_score(y2_test, y_pred_test2))
print("Case 2: Test RMSE:", mean_squared_error(y2_test, y_pred_test2) ** 0.5)
print("Case 2: Test MAE:", mean_absolute_error(y2_test, y_pred_test2))
# Evaluation for Case 3
print("\nCase 3: Training R²:", r2_score(y3_train, y_pred_train3))
print("Case 3: Test R²:", r2_score(y3_test, y_pred_test3))
print("Case 3: Test RMSE:", mean_squared_error(y3_test, y_pred_test3) ** 0.5)
print("Case 3: Test MAE:", mean_absolute_error(y3_test, y_pred_test3))
# Evaluation for Case 4
print("\nCase 4: Training R²:", r2_score(y4_train, y_pred_train4))
print("Case 4: Test R²:", r2_score(y4_test, y_pred_test4))
print("Case 4: Test RMSE:", mean_squared_error(y4_test, y_pred_test4) ** 0.5)
print("Case 4: Test MAE:", mean_absolute_error(y4_test, y_pred_test4))

results = pd.DataFrame({
    'Case': ['Age', 'Family Size', 'Age + Family Size', 'Parch'],
    'Train R²': [
        r2_score(y1_train, y_pred_train1),
        r2_score(y2_train, y_pred_train2),
        r2_score(y3_train, y_pred_train3),
        r2_score(y4_train, y_pred_train4)
    ],
    'Test R²': [
        r2_score(y1_test, y_pred_test1),
        r2_score(y2_test, y_pred_test2),
        r2_score(y3_test, y_pred_test3),
        r2_score(y4_test, y_pred_test4)
    ],
    'Test RMSE': [
        mean_squared_error(y1_test, y_pred_test1) ** 0.5,
        mean_squared_error(y2_test, y_pred_test2) ** 0.5,
        mean_squared_error(y3_test, y_pred_test3) ** 0.5,
        mean_squared_error(y4_test, y_pred_test4) ** 0.5,
    ],
    'Test MAE': [
        mean_absolute_error(y1_test, y_pred_test1),
        mean_absolute_error(y2_test, y_pred_test2),
        mean_absolute_error(y3_test, y_pred_test3),
        mean_absolute_error(y4_test, y_pred_test4)
    ]
})

print(results)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y1_test, y_pred_test1, alpha=0.6)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Case 1: Age")

plt.subplot(1, 2, 2)
plt.scatter(y3_test, y_pred_test3, alpha=0.6, color='orange')
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Case 3: Age + Family Size")

plt.tight_layout()
plt.show()
