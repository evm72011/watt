import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

folder = 'ruwatt/data'
file_name = f'{folder}/auto.csv'

df = pd.read_csv(file_name)
x, y = df.iloc[:, 1:-1], df.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Coefficients: \n", model.coef_)
print("Intercept: ", model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
