from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.DataFrame({"score": [70, 55, 90, 80, 85, 70]})
print(df)
scaler = StandardScaler()
df['score_std'] = scaler.fit_transform(df[['score']])
print(df)
print(df[['score']])