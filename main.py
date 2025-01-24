import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")

print(df)