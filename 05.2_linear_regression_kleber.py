import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

folder = 'ruwatt/data'
file_name = f'{folder}/kleibers_law.csv'

df = pd.read_csv(file_name, header=None)
df = df.map(np.log)
x = df.iloc[0].to_numpy().reshape(-1, 1)
y = df.iloc[1].to_numpy().reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Coefficients: \n", model.coef_)
print("Intercept: ", model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

x_10 = np.log(np.array([[10.0]]))
y_10 = np.exp(model.predict(x_10))[0][0]
print(f"Mass: 10 kg; Energy: {y_10} J / {y_10 / 4.18} cal")

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.scatter(x_test, y_test, color='orange', label='Test data')
plt.plot(x_test, y_pred, color='green', label='Predicted line')
plt.xlabel('Body mass')
plt.ylabel('Energy')
plt.title('Kleiber’s Metabolic rate law')
plt.legend()
plt.show()