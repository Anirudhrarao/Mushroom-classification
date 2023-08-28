import pandas as pd 

df = pd.read_csv("artifacts/data.csv")

print(df['gill_spacing'].value_counts)






