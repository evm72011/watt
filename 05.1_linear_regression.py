import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

colnames=['period', 'debt'] 
df = pd.read_csv('data/05.1_student_debt_data.csv', names=colnames, header=None)
x = np.array(df.index).reshape(-1, 1)
y = df['debt']
period = df['period']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Coefficients: \n", model.coef_)
print("Intercept: ", model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

x_2030 = len(x) + 3 + 16 * 4
y_2030 = model.predict(np.array([[x_2030]]))[0]
print(f"2030: {y_2030}")

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.scatter(x_test, y_test, color='orange', label='Test data')
plt.plot(x_test, y_pred, color='green', label='Predicted line')
plt.xlabel('Period')
plt.ylabel('Debt')
plt.title('Linear Regression on Student Debt Data')
plt.legend()
#plt.xticks(ticks=range(len(period)), labels=period, rotation=45) 
plt.show()
