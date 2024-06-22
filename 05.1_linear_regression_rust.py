import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

colnames=['period', 'debt'] 
df = pd.read_csv('ruwatt\data\student_debt_test.csv', names=colnames, header=None)
y_test = df['debt']
x_test = df['period']

df = pd.read_csv('ruwatt\data\student_debt_train.csv', names=colnames, header=None)
y_train = df['debt']
x_train = df['period']

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='orange', label='Test')
plt.xlabel('Period')
plt.ylabel('Debt')
plt.title('Linear Regression on Student Debt Data')
plt.legend()
plt.show()