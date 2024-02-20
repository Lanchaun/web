import streamlit as st
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import sys

st.sidebar.title('Variables Input')
loaded_model = lgb.Booster(model_file='lgbm_model.txt')

# # 定义预测函数
# def predict(esophageal_cancer_data, model):
#     # 使用模型进行预测
#     predictions = model.predict(esophageal_cancer_data, num_iteration=model.best_iteration)
#     return predictions
#
# # # 创建主函数
# def main():
#     # 加载模型
#     model = loaded_model()

    # 创建标题
st.title('Predicting Liver Metastasis in Esophageal Cancer')

    # 添加输入特征
st.header('Input Features')

    # 通过Streamlit的侧边栏添加输入特征
Gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
Histology = st.sidebar.selectbox('Histology', ['Adenocarcinoma', 'Others','Squamous cell carcinoma'])
Brain_metastases = st.sidebar.selectbox('Brain metastases', ['No', 'Yes'])
Lung_metastases = st.sidebar.selectbox('Lung metastases', ['No', 'Yes'])
Regional_surgery_method = st.sidebar.selectbox('Regional surgery method', ['1 to 3 regional lymph nodes removed', '4 or more regional lymph nodes removed','Biopsy or aspiration of regional lymph node, NOS','Non-surgery','Sentinel lymph node biopsy','Sentinel node biopsy and lym nd removed different times','Sentinel node biopsy and lym nd removed same/unstated time'])
Chemotherapy_recode = st.sidebar.selectbox('Chemotherapy recode', ['No/Unknown', 'Yes'])
# tumor_size = st.sidebar.number_input('肿瘤大小', min_value=0.0, max_value=20.0, value=5.0)
# 构建特征DataFrame
input_data = pd.DataFrame({
        'Gender': [Gender],
        'Histology':[Histology],
        'Brain metastases':[Brain_metastases],
        'Lung metastases':[Lung_metastases],
        'Regional surgery method':[Regional_surgery_method],
        'Chemotherapy recode':[Chemotherapy_recode]
        # '肿瘤大小': [tumor_size]
    })

    # 对性别进行编码
input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})
input_data['Histology'] = input_data['Histology'].map({'Adenocarcinoma':0, 'Others':1,'Squamous cell carcinoma':2})
input_data['Brain metastases'] = input_data['Brain metastases'].map({'No': 0, 'Yes': 1})
input_data['Lung metastases'] = input_data['Lung metastases'].map({'No': 0, 'Yes': 1})
input_data['Regional surgery method'] = input_data['Regional surgery method'].map({'1 to 3 regional lymph nodes removed':0, '4 or more regional lymph nodes removed':1,'Biopsy or aspiration of regional lymph node, NOS':2,'Non-surgery':3,'Sentinel lymph node biopsy':4,'Sentinel node biopsy and lym nd removed different times':5,'Sentinel node biopsy and lym nd removed same/unstated time':6})
input_data['Chemotherapy recode'] = input_data['Chemotherapy recode'].map({'No/Unknown':0, 'Yes':1})
    # 展示输入数据
st.write('Input Data：')
st.write(input_data)
st.write("LightGBM AUC：0.8039016")#是不是固定的值？？
#     # 如果有数据，进行预测
# if st.button('预测'):
#         # 预测
#     # predictions = loaded_model.predict(input_data)
#     lgbm_y_pred = loaded_model.predict(input_data)
#     # lgbm_y_proba = lgbm_model.predict_proba(X_test)[:, 1]
#     # lgbm_cm = confusion_matrix(y_test, lgbm_y_pred)
#     # lgbm_auc = roc_auc_score(y_test, lgbm_y_proba)
#     # lgbm_balanced_accuracy = balanced_accuracy_score(y_test, lgbm_y_pred)
#         # 显示预测结果
#     st.write('预测结果：')
#     st.write(lgbm_y_pred)
#
# # # 运行主函数
# # if __name__ == '__main__':
# #     main()

if st.button('Predict'):
    X = pd.DataFrame(input_data)
    y_pred = loaded_model.predict(X)
    # y_proba = loaded_model.predict_proba(X）
    # auc = roc_auc_score(y_test, y_proba)
    #
    # if y_pred[0]:
    #     st.success('该病人有食管癌肝转移的风险，AUC值为{}'.format(auc))
    # else:
    #     st.success('该病人无食管癌肝转移的风险，AUC值为{}'.format(auc))
    st.write('Probability of liver metastases：')
    st.write(y_pred)