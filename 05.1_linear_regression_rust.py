import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

'''
Run the Rust integration test linear_regression_students
'''

PERIOD = 'period'
DEBT = 'debt'

colnames=[PERIOD, DEBT] 
folder = 'ruwatt/data/results/student_debt'
df = pd.read_csv(f'{folder}/test.csv', names=colnames, header=None)
y_test = df[DEBT]
x_test = df[PERIOD]

df = pd.read_csv(f'{folder}/train.csv', names=colnames, header=None)
y_train = df[DEBT]
x_train = df[PERIOD]

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='orange', label='Test')
plt.xlabel('Period')
plt.ylabel('Debt')
plt.title('Linear Regression on Student Debt Data')
plt.legend()
plt.show()