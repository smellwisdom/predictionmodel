import streamlit as st
import shap
import pandas as pd
import joblib
import numpy as np
import streamlit.components.v1 as components
from scipy.special import expit  # 用于sigmoid函数
import os

# 确认模型文件路径
model_path = 'GBDT.pkl'  # 请根据需要调整路径

# 确保模型文件路径正确
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    # 加载训练好的模型
    loaded_model = joblib.load(model_path)

    # 特征名称
    feature_names = ['Age', 'Admission_mRS_Score', 'Thrombolytic_Drug', 'PreTreat_ASPECT_Score',
                     'Onset_to_Puncture_Time', 'Antiplatelet_Therapy', 'Anticoagulation_Therapy']

    # Streamlit 应用程序接口
    st.title("Patient HI Prediction")

    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Input Patient Details:</p>', unsafe_allow_html=True)

    # 创建输入表单并添加合理的限制
    input_data = {}
    input_data['Age'] = st.number_input('Age', min_value=0.0, max_value=120.0, value=60.0, step=1.0)
    input_data['Admission_mRS_Score'] = st.number_input('Admission mRS Score', min_value=0.0, max_value=6.0, value=3.0, step=0.5)
    input_data['Thrombolytic_Drug'] = st.selectbox('Thrombolytic Drug', options=['rt-pa', 'Other'], index=0)
    input_data['PreTreat_ASPECT_Score'] = st.number_input('PreTreat ASPECT Score', min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    input_data['Onset_to_Puncture_Time'] = st.number_input('Onset to Puncture Time (min)', min_value=0.0, max_value=300.0, value=20.0, step=1.0)
    input_data['Antiplatelet_Therapy'] = st.selectbox('Antiplatelet Therapy', options=[0, 1], index=1)
    input_data['Anticoagulation_Therapy'] = st.selectbox('Anticoagulation Therapy', options=[0, 1], index=1)

    # 将输入数据转换为 DataFrame
    input_df = pd.DataFrame([input_data])

    # 将 Thrombolytic_Drug 转换为数值
    input_df['Thrombolytic_Drug'] = input_df['Thrombolytic_Drug'].map({'rt-pa': 1, 'Other': 0})

    # 进行预测
    if st.button('Predict'):
        prediction_prob = loaded_model.predict_proba(input_df)[0, 1]
        st.markdown('<p class="big-font">Prediction Result:</p>', unsafe_allow_html=True)
        st.write(f"Based on feature values, predicted probability of HI is {prediction_prob * 100:.2f}%")

        # 创建 SHAP 解释器
        explainer = shap.Explainer(loaded_model, feature_names=feature_names)
        shap_values = explainer(input_df)

        # 如果基线值是标量，使用它，否则选择索引0或1
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray) and len(base_value) > 1:
            base_value = base_value[0]

        # 创建 Force Plot 并保存为HTML文件
        shap.initjs()
        force_plot = shap.force_plot(base_value, shap_values.values[0], input_df, feature_names=feature_names, link='logit')

        # 转换SHAP值为概率值
        prob_value = expit(base_value + shap_values.values[0].sum())

        shap.save_html("force_plot.html", force_plot)
        
        # 读取HTML文件并在Streamlit中显示
        with open("force_plot.html", "r", encoding="utf-8") as f:
            force_plot_html = f.read()

        # 使用 Streamlit 显示 HTML 力图，并使其居中显示
        st.markdown('<div class="centered">', unsafe_allow_html=True)
        components.html(force_plot_html, height=800, width=1200)  # 调整高度和宽度确保显示完全
        st.markdown('</div>', unsafe_allow_html=True)
