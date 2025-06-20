import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

# 🚀 OpenAI istemcisi (API KEY ortamdan alınıyor)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🚀 Flask app
app = Flask(__name__)
CORS(app)

# 🚀 Veri yükle ve modeli eğit
DATA_FILE = 'cart.csv'
data = pd.read_csv(DATA_FILE)

X = data[['price', 'session_duration', 'pages_viewed', 'returning_user', 'discount_shown']]
y = data['purchase_made']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 🚀 Tahmin endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame([input_data])
    probability = model.predict_proba(input_df)[0][1]
    return jsonify({'purchase_probability': probability})

# 🚀 ChatGPT tavsiye endpoint'i
@app.route('/advice', methods=['POST'])
def advice():
    try:
        data = request.json
        prob = data.get('probability')
        params = data.get('parameters')

        # Güvenlik: Eksik veri kontrolü
        if prob is None or params is None:
            return jsonify({"error": "Missing probability or parameters"}), 400

        prompt = f"""
        The current purchase probability is {prob:.2%}.
        Customer info: {params}.
        Give a short marketing suggestion to increase the purchase probability.
        """

        # En garantili model: gpt-3.5-turbo
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful marketing advisor AI."},
                {"role": "user", "content": prompt}
            ]
        )

        advice_text = response.choices[0].message.content.strip()
        return jsonify({"advice": advice_text})

    except Exception as e:
        # Loglama için: hata detayını da göster
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)