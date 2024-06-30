import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def normalize(x):
    mean = np.mean(x, axis=0)
    std_dev = np.std(x, axis=0)
    return (x - mean) / std_dev

folder = 'ruwatt/data'
file_name = f'{folder}/boston_housing.csv'

df = pd.read_csv(file_name)
X, y = df.drop('medv', axis=1), df['medv']
X, y = normalize(X), normalize(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Coefficients: \n", model.coef_)
print("Intercept: ", model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
