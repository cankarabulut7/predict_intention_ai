import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Predictive İntention AI - Gelişmiş Sepet Terk Tahmini Paneli")

uploaded_file = st.file_uploader("CSV Dosyası Yükle", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    features = ['price', 'session_duration', 'pages_viewed', 'returning_user', 'discount_shown']

    X = df[features]
    y = df['purchased']

    model = LogisticRegression(max_iter=1000).fit(X, y)

    df['purchase_probability'] = model.predict_proba(X)[:, 1]

    st.write(df)

    st.line_chart(df['purchase_probability'])