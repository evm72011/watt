import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

folder = 'ruwatt/data'
file_name = f'{folder}/breast_cancer_wisconsin.csv'

df = pd.read_csv(file_name)
print(df.columns.tolist())
df.drop('id')