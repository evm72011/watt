import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

'''
Run the Rust integration test linear_regression_kleiber
'''

folder = 'ruwatt/data/results/kleibers_law'
df = pd.read_csv(f'{folder}/test.csv', header=None)
x_test, y_test = df[0], df[1]

df = pd.read_csv(f'{folder}/train.csv', header=None)
x_train, y_train = df[0], df[1]

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='orange', label='Test')
plt.xlabel('Body mass')
plt.ylabel('Energy')
plt.title('Kleiber’s Metabolic rate law')
plt.legend()
plt.show()