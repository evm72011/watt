import pandas as pd
import numpy as np

folder = 'ruwatt/data'
file_name = f'{folder}/kleibers_law.csv'

df = pd.read_csv(file_name, header=None)
data = np.loadtxt(file_name, delimiter=',')
x = data[:-1,:]
y = data[-1:,:]

print(np.shape(x))
print(np.shape(y))