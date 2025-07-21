import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
# 设置字体，防止中文显示为乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号


data_train = pd.read_csv('train.csv')
# print(data_train)
# print(data_train.info)
# print(data_train.describe())
#
# fig = plt.figure(figsize=(16, 10))
# fig.set(alpha=0.2)
# plt.subplot2grid((2, 3), (0, 0))
# survived_counts = data_train.Survived.value_counts().sort_index()
# survived_counts.plot(kind='bar')
# plt.xticks([0, 1], ['未获救', '获救'], rotation=0)
# plt.title(u"获救情况")
# plt.ylabel(u"人数")
#
# plt.subplot2grid((2, 3), (0, 1))
# Pclass_counts = data_train.Pclass.value_counts().sort_index()
# Pclass_counts.plot(kind='bar')
# plt.grid(True, linestyle='--', axis='y')  # 正确的位置和语法
# plt.xticks([0, 1, 2], ['头等舱', '2等舱', '3等舱'], rotation=0)
# plt.title(u"乘客等级分布")
# plt.ylabel(u"人数")
#
# #柱状图
# plt.subplot2grid((2, 3), (0, 2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"年龄")
# #图表设计
# plt.grid(visible=True, which='major', axis='y', linestyle='--', alpha=0.7)
# #标签替换
# plt.xticks([0, 1], ['获救', '未获救'], rotation=0)
# plt.title(u"不同年龄获救分布")
#
#
# #曲线图
# plt.subplot2grid((2, 3), (1, 0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# # 绘制三等舱乘客年龄的核密度估计曲线
# plt.xlabel(u"年龄")  # x轴标签
# plt.ylabel(u"密度")  # y轴标签，表示概率密度
# plt.title(u"各等级的乘客年龄分布")  # 图表标题
# plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')
#
# plt.subplot2grid((2, 3), (1, 2))
# Embarked_train = data_train.Embarked.value_counts()
# Embarked_train.plot(kind='bar')
# plt.ylabel(u"人数")
# plt.title(u"各登船口上岸人数")
#
# plt.show()
#
# #不同经济地位获救情况统计
# fig1 = plt.figure(figsize=(16, 10))
# fig1.set(alpha=0.2)
#
# Survived0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived1 = data_train.Pclass[data_train.Survived == 1].value_counts()
#
# df = pd.DataFrame({u"获救": Survived0, u"未获就": Survived1})
#
# df.plot(kind='bar', stacked=True)
# plt.title(u"乘客各等级的获救情况")
# plt.xticks([0, 1, 2], ['头等舱', '2等舱', '3等舱'], rotation=0)
# plt.xlabel(u"乘客等级")
# plt.ylabel(u"获救人数")
# plt.show()
#
#
# # 查看不同登船港口乘客的获救和未获救人数分布情况
#
# fig2 = plt.figure()
# fig2.set(alpha=0.2)  # 设置图表整体透明度，使颜色更柔和，不刺眼
#
# # 统计未获救乘客中，不同登船港口的人数
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
#
# # 统计获救乘客中，不同登船港口的人数
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
#
# # 将获救和未获救的人数合并成一个DataFrame，方便绘图对比
# df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
#
# # 绘制堆叠柱状图，显示不同登船港口中获救和未获救人数的分布情况
# df.plot(kind='bar', stacked=True)
#
# # 添加图表标题
# plt.title(u"各登录港口乘客的获救情况")
#
# # 设置x轴标签，表示不同的登船港口
# plt.xlabel(u"登录港口")
#
# # 设置y轴标签，表示人数
# plt.ylabel(u"人数")
#
# # 显示图表
# plt.show()
#
# # 查看不同性别乘客的获救和未获救人数分布情况
# fig3 = plt.figure()
# fig3.set(alpha=0.2)  # 设置图表整体透明度，让颜色看起来更柔和
#
# # 统计男性乘客中获救和未获救的人数
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
#
# # 统计女性乘客中获救和未获救的人数
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
#
# # 将男性和女性乘客的获救情况合并成一个DataFrame，方便绘图对比
# df = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
#
# # 绘制堆叠柱状图，展示不同性别乘客的获救和未获救人数分布
# df.plot(kind='bar', stacked=True)
#
# # 添加图表标题，说明展示的是按性别划分的获救情况
# plt.title(u"按性别看获救情况")
# plt.xticks([0, 1], [u"男性", u"女性"], rotation=0)
# # 设置x轴标签，表示性别
# plt.xlabel(u"性别")
# # 设置y轴标签，表示人数
# plt.ylabel(u"人数")
# # 显示图表
# plt.show()
#
#
# fig = plt.figure(figsize=(10, 6))
# fig.set(alpha=0.65)
#
# plt.suptitle(u"根据船舱等级和性别的获救情况", fontsize=16)
# ax1 = fig.add_subplot(221)
# data_train.Survived[(data_train.Sex == 'female') & (data_train.Pclass != 3)].value_counts().plot(
#     kind='bar', color='#FA2479', ax=ax1)
# ax1.set_xticklabels([u"未获救", u"获救"], rotation=0)
# ax1.set_title(u"女性 / 高级仓")
# ax1.set_ylabel(u"人数")
#
# ax2 = fig.add_subplot(222, sharey=ax1)
# data_train.Survived[(data_train.Sex == 'female') & (data_train.Pclass == 3)].value_counts().plot(
#     kind='bar', color='pink', ax=ax2)
#
# ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
# ax2.set_title(u"女性 / 低级仓")
#
#
# ax3 = fig.add_subplot(223, sharey=ax1)
# data_train.Survived[(data_train.Sex == 'male') & (data_train.Pclass != 3)].value_counts().plot(
#     kind='bar', color='lightblue', ax=ax3)
#
# ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
# ax3.set_title(u"男性 / 低级仓")
# ax3.set_ylabel(u"人数")
#
# ax4 = fig.add_subplot(224, sharey=ax1)
# data_train.Survived[(data_train.Sex == 'male') & (data_train.Pclass == 3)].value_counts().plot(
#     kind='bar', color='steelblue', ax=ax4
# )
# ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
# ax4.set_title(u"男性 / 低级仓")
#
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 自动调整子图间距，避免标题遮挡
# plt.show()


# print("Cabin 非空数量: ", data_train['Cabin'].notnull().sum())
# print("Cabin 空数量: ", data_train['Cabin'].isnull().sum())

#随机深林填充缺失值
from sklearn.ensemble import RandomForestRegressor


def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    know_df = age_df[age_df.Age.notnull()].to_numpy()
    unknow_df = age_df[age_df.Age.isnull()].to_numpy()

    y = know_df[:, 0]
    X = know_df[:, 1:]

    rft = RandomForestRegressor(random_state=0, n_estimators=1000, n_jobs=-1)
    rft.fit(X, y)

    predicted_age = rft.predict(unknow_df[:, 1:])

    df.loc[df.Age.isnull(), 'Age'] = predicted_age
    df['Age'] = df['Age'].round().astype(int)
    return df, rft



data_train, rft0 = set_missing_ages(data_train)
print(data_train)



#热编码
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked', dtype=int)

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex', dtype=int)

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass', dtype=int)

df = pd.concat([data_train, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Embarked'], axis=1, inplace=True)
print(df)


import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()

age_scaled = scaler.fit(df[['Age']])
df['Age_scaled'] = scaler.transform(df[['Age']])


fare_scale_param = scaler.fit(df[['Fare']])
df['Fare_scaled'] = scaler.transform(df[['Fare']])
print(df)


from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 从预处理过的 df 中筛选出用于建模的特征字段
# 使用正则表达式选择列名中包含如下字段的列：
# - Survived：作为目标变量（标签）
# - Age_.*：即 Age_scaled，表示归一化后的年龄
# - SibSp：兄弟姐妹/配偶数，原始特征，数值型
# - Parch：父母/子女数，原始特征，数值型
# - Fare_.*：即 Fare_scaled，表示归一化后的票价
# - Embarked_.*：登船港口的 one-hot 编码列
# - Sex_.*：性别的 one-hot 编码列
# - Pclass_.*：舱位等级的 one-hot 编码列

# 1. 筛选训练数据的特征和标签
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
data_np = train_df.to_numpy()
y = data_np[:, 0]
X = data_np[:, 1:]

# 2. 划分训练集和测试集（测试集占30%，随机种子固定，方便复现）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 初始化逻辑回归模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6, solver='liblinear')

# 4. 用训练集训练模型
clf.fit(X_train, y_train)

# 5. 用测试集预测
y_pred = clf.predict(X_test)

# 6. 计算并输出测试集的指标
print("测试集准确率 (Accuracy): {:.4f}".format(accuracy_score(y_test, y_pred)))
print("测试集精确率 (Precision): {:.4f}".format(precision_score(y_test, y_pred)))
print("测试集召回率 (Recall): {:.4f}".format(recall_score(y_test, y_pred)))
print("测试集F1分数 (F1 Score): {:.4f}".format(f1_score(y_test, y_pred)))