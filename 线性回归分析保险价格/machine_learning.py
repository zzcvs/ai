import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('insurance.csv')
# print(df.head())

print(df.dtypes)

# print(df.isnull().sum())

sns.set_style(style='whitegrid')
f, ax = plt.subplots(figsize=(12, 8))
sns.histplot(data=df, x='charges', kde=True, color='c', ax=ax)
plt.title("Charges distribution")
# plt.show()

charges = df['charges'].groupby(df.region).sum()

charges = charges.sort_values(ascending=True)

f, ax = plt.subplots(1, 1, figsize=(12, 8))

sns.barplot(x=charges.values[:5], y=charges.index[:5], palette='Blues', ax=ax)

#给绘制的图表添加标题，说明图的含义
plt.title('Total Charges by Region (Top 5)')
plt.show()

f, ax = plt.subplots(1, 1, figsize=(12, 8))  # 创建一个12x8英寸大小的图形和子图
ax = sns.barplot(x='region', y='charges', hue='sex', data=df, palette='cool')
# 绘制按地区(region)分组的医疗费用(charges)条形图，按照性别(sex)分色显示，使用cool色系调色板
plt.show()