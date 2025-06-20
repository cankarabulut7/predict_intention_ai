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
    input_data = request.json
    probability = input_data.get('probability')
    parameters = input_data.get('parameters')

    prompt = f"""
    The user is about to purchase. Predicted probability is {probability:.2%}.
    User details: {parameters}.
    Give a short smart advice to increase the chance.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a smart marketing AI."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
    return jsonify({'advice': answer})

# 🚀 Çalıştır
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)