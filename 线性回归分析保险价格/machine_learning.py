import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('insurance.csv')
# print(df.head())

# print(df.dtypes)

# print(df.isnull().sum())
# 1
sns.set_style(style='whitegrid')
f, ax = plt.subplots(figsize=(12, 8))
sns.histplot(data=df, x='charges', kde=True, color='c', ax=ax)
plt.title("Charges distribution")
# plt.show()

charges = df['charges'].groupby(df.region).sum()

charges = charges.sort_values(ascending=True)
#2
f, ax = plt.subplots(1, 1, figsize=(12, 8))
sns.barplot(x=charges.values[:5], y=charges.index[:5], palette='Blues', ax=ax)
#给绘制的图表添加标题，说明图的含义
plt.title('Total Charges by Region (Top 5)')
# plt.show()

# 3
f, ax = plt.subplots(1, 1, figsize=(12, 8))  # 创建一个12x8英寸大小的图形和子图
ax = sns.barplot(x='region', y='charges', hue='sex', data=df, palette='cool')
# 绘制按地区(region)分组的医疗费用(charges)条形图，按照性别(sex)分色显示，使用cool色系调色板
# plt.show()

# 4
f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.barplot(x='region', y='charges', hue='smoker', data=df, palette='Reds_r')
# plt.show()

# 5
f, ax = plt.subplots(1, 1, figsize=(12, 8))  # 创建一个12x8英寸大小的图形和子图

ax = sns.barplot(
    x='region',        # x轴表示地区(region)
    y='charges',       # y轴表示医疗费用(charges)
    hue='children',    # 根据有无子女(children)分组显示不同颜色
    data=df,           # 使用的数据是DataFrame df
    palette='Set1'     # 使用Set1调色板，颜色鲜明，方便区分
)
# plt.show()

#6 7 8
# 根据年龄（age）和是否吸烟（smoker）绘制线性回归图，观察年龄与医疗费用之间的关系
ax = sns.lmplot(x='age', y='charges', hue='smoker', data=df, palette='Set1')

# 根据体质指数（bmi）和是否吸烟（smoker）绘制线性回归图，查看bmi与医疗费用之间的关系
ax = sns.lmplot(x='bmi', y='charges', data=df, hue='smoker', palette='Set2')
# 根据子女数量（children）和是否吸烟（smoker）绘制线性回归图，分析抚养子女数量与医疗费用的关系
ax = sns.lmplot(x='children', y='charges', data=df, hue='smoker', palette='Set3')
# plt.show()

#9
f, ax = plt.subplots(1, 1, figsize=(10, 10))
# 创建一个 10x10 英寸的画布，并返回一个图形对象（f）和子图对象（ax）

ax = sns.violinplot(
    x='children',              # 横轴为子女数量
    y='charges',               # 纵轴为医疗费用
    data=df,                   # 使用的数据集为 df
    orient='v',                # 垂直方向绘制（v 表示 vertical）
    hue='smoker',       # 按是否吸烟分类绘图，不同颜色代表吸烟与否
    palette='inferno'          # 使用 inferno 颜色调色板（从亮黄到暗红）
)
# plt.show()
# 使用小提琴图展示不同子女数量下，不同吸烟状态人群的医疗费用分布情况
# 小提琴图结合了箱线图和核密度图，能够更直观地显示数据的分布趋势和密度


## 将对象类型的标签转换为分类类型
df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype('category')
# 将数据框 df 中的 'sex'（性别）、'smoker'（是否吸烟）、'region'（地区）三列的数据类型
# 从 object（字符串）转换为 category（分类）类型，这样更节省内存，并利于后续分析和建模

## 使用 LabelEncoder 将类别标签转换为数值（编码）
from sklearn.preprocessing import LabelEncoder  # 从 sklearn 中导入 LabelEncoder，用于标签编码

label = LabelEncoder()  # 创建一个 LabelEncoder 对象，用于将字符串类型的分类变量转换为整数编码
label.fit(df.sex.drop_duplicates())
# 去除 sex 列中的重复值，并使用这些唯一值来训练编码器，即建立“性别-数值”的映射关系
df.sex = label.transform(df.sex)
# 将 sex 列中的所有值按照上面建立的映射关系转换为数值
label.fit(df.smoker.drop_duplicates())
# 同样地，对 smoker（是否吸烟）列做去重并训练编码器
df.smoker = label.transform(df.smoker)
# 将 smoker 列转换为数值
label.fit(df.region.drop_duplicates())
# 对 region（地区）列做去重并训练编码器
df.region = label.transform(df.region)
# 将 region 列转换为数值
# print(df.dtypes)

f, ax = plt.subplots(1, 1, figsize=(10, 10))  # 创建一个10x10英寸的图像窗口和一个子图对象 ax

ax = sns.heatmap(df.corr(), annot=True, cmap='cool')


from sklearn.model_selection import train_test_split as holdout
# 导入train_test_split函数，并重命名为holdout，用于将数据分为训练集和测试集
# 导入线性回归模型类
from sklearn.linear_model import LinearRegression

# 导入评估指标模块
from sklearn import metrics
# 将数据集中的特征变量赋值给x，去掉目标变量'charges'
x = df.drop(['charges'], axis=1)

# 将目标变量赋值给y，这里是医疗费用
y = df['charges']

# 将数据集划分为训练集和测试集，测试集占20%
# random_state=0保证每次划分结果相同，方便复现
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
# 创建一个线性回归模型实例
Lin_reg = LinearRegression()

# 用训练集数据训练线性回归模型，拟合模型参数
Lin_reg.fit(x_train, y_train)

# 输出线性回归模型的各个特征对应的系数（权重）
print(Lin_reg.coef_)

from sklearn.linear_model import Lasso
# 导入Lasso回归模型类

Lasso = Lasso(alpha=0.2)
# 创建Lasso回归模型实例，设置正则化参数alpha=0.2

Lasso.fit(x_train, y_train)
# 用训练集数据训练Lasso回归模型，拟合参数

print(Lasso.coef_)
# 输出Lasso模型的各特征系数（权重），部分系数可能被压缩为0，实现特征选择

print(Lasso.score(x_test, y_test))
# 输出模型在测试集上的R²决定系数，评价拟合效果，越接近1越好
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

x = df.drop(['charges', 'sex','region'], axis=1)
# 去掉不需要的特征 'charges'(目标变量), 'sex' 和 'region'，只保留数值型变量作为输入特征

y = df.charges
# 目标变量为 'charges'

pol = PolynomialFeatures(degree=2)
# 创建多项式特征生成器，degree=2表示生成二次多项式特征（包括原始特征的平方项和交叉项）

x_pol = pol.fit_transform(x)
# 将原始特征转换为多项式特征矩阵

x_train, x_test, y_train, y_test = holdout(x_pol, y, test_size=0.2, random_state=0)
# 将数据划分为训练集和测试集，测试集占20%，随机种子固定为0保证结果可复现

Pol_reg = LinearRegression()
# 创建线性回归模型实例

Pol_reg.fit(x_train, y_train)
# 用训练数据拟合多项式回归模型

y_test_pred = Pol_reg.predict(x_test)
r2 = r2_score(y_test, y_test_pred)
print("R2 score:", r2)


from sklearn.ensemble import RandomForestRegressor as rfr
# 导入随机森林回归模型，并简写为rfr

# 将数据集中的特征变量赋值给x，去掉目标变量'charges'
x = df.drop(['charges'], axis=1)

# 将目标变量赋值给y，这里是医疗费用
y = df['charges']

# 将数据集划分为训练集和测试集，测试集占20%
# random_state=0保证每次划分结果相同，方便复现
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)

Rfr = rfr(n_estimators=100, criterion='squared_error', random_state=1, n_jobs=-1)
# 创建随机森林回归模型实例
# n_estimators=100 表示用100棵树
# criterion='squared_error' 表示用均方误差
# random_state=1 保证结果可复现
# n_jobs=-1 表示使用所有CPU核进行训练，加速计算

Rfr.fit(x_train, y_train)
# 训练随机森林模型

x_train_pred = Rfr.predict(x_train)
# 预测训练集的结果

x_test_pred = Rfr.predict(x_test)
# 预测测试集的结果
print('R2 train data: %.3f, R2 test data: %.3f' %
      (metrics.r2_score(y_train, x_train_pred),
       metrics.r2_score(y_test, x_test_pred)))
# 打印训练集和测试集的R²决定系数，越接近1模型越好

plt.figure(figsize=(8,6))
# 创建一个大小为8x6英寸的图形窗口

plt.scatter(x_train_pred, x_train_pred - y_train,
          c='gray', marker='o', s=35, alpha=0.5,
          label='Train data')
# 绘制训练数据的残差散点图
# 横坐标是预测值，纵坐标是预测值减去真实值（残差）
# 点的颜色是灰色，形状是圆点，大小是35，透明度为0.5，标签为“Train data”
plt.scatter(x_test_pred, x_test_pred - y_test,
          c='blue', marker='o', s=35, alpha=0.7,
          label='Test data')
# 绘制测试数据的残差散点图
# 点的颜色是蓝色，透明度为0.7，标签为“Test data”
plt.xlabel('Predicted values')
# 设置x轴标签为“Predicted values”（预测值）

plt.ylabel('Actual values')
# 设置y轴标签为“Actual values”（实际值）——这里其实是残差，应该是残差值

plt.legend(loc='upper right')
# 显示图例，位置在右上角

plt.hlines(y=0, xmin=0, xmax=60000, lw=2, color='red')
# 在y=0位置画一条红色的水平线，宽度为2，横跨x轴0到60000的范围
# 这条线表示残差为0，即预测值等于实际值的位置

print('Feature importance ranking\n\n')
# 打印标题，表示开始输出特征重要性排名

importances = Rfr.feature_importances_
# 获取随机森林模型中每个特征的重要性分数

std = np.std([tree.feature_importances_ for tree in Rfr.estimators_], axis=0)
# 计算所有决策树中特征重要性的标准差，用于误差条显示

indices = np.argsort(importances)[::-1]
# 按特征重要性从高到低排序，得到特征索引的降序排列

variables = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
# 定义特征名称列表，顺序对应输入数据的列顺序

importance_list = []
# 创建一个空列表，用来存储排序后的特征名称

for f in range(x.shape[1]):
    variable = variables[indices[f]]  # 根据排序索引取出对应的特征名称
    importance_list.append(variable)  # 加入列表
    print("%d.%s(%f)" % (f + 1, variable, importances[indices[f]]))
    # 打印特征排名，格式为：序号.特征名(重要性分数)

# 绘制特征重要性柱状图
plt.figure()
plt.title("Feature importances")  # 设置图表标题
plt.bar(importance_list, importances[indices],
       color="y", yerr=std[indices], align="center")
# 横坐标为特征名称，纵坐标为重要性分数
# 柱状图颜色为黄色，误差条为标准差，柱子居中对齐
plt.show()