from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 💡 OpenAI istemcisi
from openai import OpenAI

# 🔑 Ortam değişkeninden API anahtarı kullan
import os

client = OpenAI()

app = Flask(__name__)
CORS(app)

# --- Model yükleme ---
DATA_FILE = 'cart.csv'
data = pd.read_csv(DATA_FILE)

X = data[['price', 'session_duration', 'pages_viewed', 'returning_user', 'discount_shown']]
y = data['purchase_made']  # 0 veya 1

model = LogisticRegression()
model.fit(X, y)


# --- Tahmin uç noktası ---
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame([input_data])
    prediction = model.predict_proba(input_df)[0][1]
    return jsonify({'purchase_probability': prediction})


# --- ChatGPT tavsiye uç noktası ---
@app.route('/advice', methods=['POST'])
def advice():
    input_data = request.json
    probability = input_data.get('probability')
    parameters = input_data.get('parameters')

    # ChatGPT çağrısı
    messages = [
        {
            "role": "system",
            "content": "Sen bir pazarlama danışmanısın. Kullanıcının alışveriş tamamlama olasılığına göre en iyi kampanya ve mesaj önerisini ver."
        },
        {
            "role": "user",
            "content": f"Kullanıcının alışveriş tamamlama olasılığı: %{round(probability*100, 2)}. Parametreler: {parameters}. Bu kullanıcının sepetini tamamlaması için en iyi strateji nedir? Kısa, net öner."
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    gpt_reply = response.choices[0].message.content
    return jsonify({'advice': gpt_reply})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)