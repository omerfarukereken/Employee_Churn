import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, classification_report
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import datetime
import warnings
import pickle
warnings.filterwarnings('ignore')

html_temp = """
<div style="background-color:Blue;padding:1.5px">
<h1 style="color:white;text-align:center;">Çalışan Ayrılma Durumu Sorgula</h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

model = pickle.load(open("orman_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

satisfaction = st.slider("Memnuniyet Seviyesi", 0, 100, 1) / 100
last_evaluation = st.slider("Son Değerlendirme", 0, 100, 1) / 100
number_project = st.slider("Proje Sayısı", 0, 20, 1)
monthly_hours = st.slider("Aylık Ortalama Çalışma Süresi", 0,500,1)
time_spend = st.slider("Şirkette Geçirilen Süre", 0,20,1)
work_accident = st.selectbox("İş Kazası", ["Hayır", "Evet"])
if work_accident == "Evet": work_accident = 1
else: work_accident = 0

promotion = st.selectbox("Son 5 yılda terfi", ["Hayır", "Evet"])
if promotion == "Evet": promotion = 1
else: promotion = 0

departments = st.selectbox("Departman", ['IT', 'RandD', 'Accounting', 'Hr', 'Management', 'Marketing', 'Product_mng', 'Sales', 'Support', 'Technical'])
salary = st.selectbox("Maaş Durumu", ["Düşük", "Orta", "Yüksek"])

df = pd.DataFrame(data=[[satisfaction, last_evaluation, number_project, monthly_hours, time_spend, work_accident, promotion]],
                  columns=['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident','promotion_last_5years'])

df_department = pd.DataFrame(data=[[0] * 10], columns=['IT', 'RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng', 'sales', 'support', 'technical'])

for i in df_department.columns:
    if i.title() == departments:
        df_department.loc[0,i] = 1

a_s = scaler.transform(df)
df2 = pd.DataFrame(a_s, columns=df.columns)
df2["salary"] = salary
df2['salary'] = df2['salary'].apply(lambda x: ["Düşük", "Orta", "Yüksek"].index(x))
df3 = df2.join(df_department)

if st.button("Tahmin"):
    prediction = model.predict(df3)[0]
    if prediction == 0:
        st.success("Kalacak")
    else:
        st.error("Gidecek")