import pandas as pd
import streamlit as st
import joblib

st.title("🫀 心脏病预测系统（随机森林）")
st.sidebar.header("输入病人信息")

def user_input_features():
    age = st.sidebar.slider("年龄", 20, 90, 50)
    sex = st.sidebar.selectbox("性别", ["male", "female"])
    cp = st.sidebar.selectbox("胸痛类型", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
    trestbps = st.sidebar.slider("静息血压", 80, 200, 130)
    chol = st.sidebar.slider("胆固醇", 100, 400, 250)
    fbs = st.sidebar.selectbox("空腹血糖 > 120mg/dl", ["false", "true"])
    restecg = st.sidebar.selectbox("静息心电图结果", ["normal", "st-t wave abnormality", "left ventricular hypertrophy"])
    thalach = st.sidebar.slider("最大心率", 60, 220, 150)
    exang = st.sidebar.selectbox("运动诱发心绞痛", ["no", "yes"])
    oldpeak = st.sidebar.slider("ST 抑制值", 0.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("ST 斜率类型", ["upsloping", "flat", "downsloping"])
    ca = st.sidebar.slider("主要血管数量", 0, 3, 0)
    thal = st.sidebar.selectbox("地中海贫血类型", ["normal", "fixed defect", "reversible defect"])

    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 加载模型和字典
model = joblib.load("model.pkl")
le_dict = joblib.load("le_dict.pkl")
feature_names = joblib.load("features.pkl")

# 编码输入
label_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
for col in label_cols:
    input_df[col] = input_df[col].astype(str).str.lower()
    input_df[col] = le_dict[col].transform(input_df[col])

# 对齐列顺序
input_df = input_df[feature_names]

# 展示输入
st.subheader("输入信息")
st.write(input_df)

# 模型预测
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]
st.subheader("预测结果")
st.write("🧠 有心脏病风险" if prediction == 1 else "✅ 未发现明显心脏病风险")
st.metric("预测概率", f"{proba*100:.2f}%")
