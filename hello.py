#这是训练好的模型，等会用cat model带入节省时间（如何将模型与实时输入的数据衔接）
import os
print(os.getcwd())  # 获取当前工作目录
from pyecharts.charts import Map
from pyecharts import options as opts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_excel('Liver3.xlsx')
import pandas as pd
df.head()
#分类变量数值化
from sklearn.preprocessing import LabelEncoder
df.iloc[:,-1] = LabelEncoder().fit_transform(df.iloc[:,-1])
from sklearn.preprocessing import OrdinalEncoder
#接口categories_对应LabelEncoder的接口classes_，一模一样的功能
df_ = df.copy()#对df进行复制，怕弄错
df_.head()
OrdinalEncoder().fit(df_.iloc[:,0:-1]).categories_#第0列是性别，到-1不包括-1
df_.iloc[:,0:-1] = OrdinalEncoder().fit_transform(df_.iloc[:,0:-1])
df_.head()
#KNN插补
#保险起见,复制一下knn的数据集
df_knn = df_.copy()
#载入包
from sklearn.impute import KNNImputer
import pandas as pd
# 创建一个包含缺失值的示例 DataFrame
df_knn_= pd.DataFrame(df_knn)#只考虑列的影响
# 实例化 KNN 填充器
imputer = KNNImputer(n_neighbors=5)#选择了5个邻居
# 使用 fit_transform 进行填充
df_knn_filled = pd.DataFrame(imputer.fit_transform(df_knn_), columns=df.columns)
# 打印填充后的 DataFrame
print(df_knn_filled)
#查看knn填充后是否填完:全都填完了
df_knn_filled.info()
#df_knn_filled 用knn填充之后的结果

df_encoded=df_knn_filled#代换一下，现在不用独热了
#%%
#定义xY
y=df_encoded['Liver#metastases']
X=df_encoded.drop(columns=['Liver#metastases'])
y.shape,X.shape

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
y_train.shape,X_train.shape

y_test.value_counts()
#对训练集过采样
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
X_train_resampled.shape, y_train_resampled.shape

y_train_resampled.value_counts()

#嵌入法
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
base_model=LogisticRegression(solver='liblinear')#增加迭代次数，需要更多的时间才能跑出来
selector=SelectFromModel(estimator=base_model)

X_train=selector.fit_transform(X_train_resampled,y_train_resampled)
X_test=selector.transform(X_test)
feature_names=X.columns[np.where(selector.get_support())]
X_train=pd.DataFrame(columns=feature_names,data=X_train)
X_test=pd.DataFrame(columns=feature_names,data=X_test)
X_train

#建模
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix,roc_curve
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train_resampled)
lgbm_model.booster_.save_model('lgbm_model.txt')

#混淆矩阵
lgbm_y_pred = lgbm_model.predict(X_test)
lgbm_y_proba = lgbm_model.predict_proba(X_test)[:, 1]
lgbm_cm = confusion_matrix(y_test, lgbm_y_pred)
lgbm_auc = roc_auc_score(y_test, lgbm_y_proba)
lgbm_balanced_accuracy = balanced_accuracy_score(y_test, lgbm_y_pred)

#输出结果
print("LightGBM 预测结果：", lgbm_y_pred)
print("LightGBM 混淆矩阵：")
print(lgbm_cm)
print("LightGBM ROC AUC 分数：", lgbm_auc)
print("LightGBM 平衡准确率：", lgbm_balanced_accuracy)

#分类报告
print("LightGBM 分类报告:")
print(classification_report(y_test, lgbm_model.predict(X_test)))



