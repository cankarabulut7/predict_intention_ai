# intention_ai.py

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# --- 1️⃣ Yeni veri dosyasını yükle ---
DATA_FILE = 'cart.csv'

# Veriyi oku
data = pd.read_csv(DATA_FILE)

# Özellikler ve etiket
X = data[['price', 'session_duration', 'pages_viewed', 'returning_user', 'discount_shown']]
y = data['purchase_probability']

# --- 2️⃣ Modeli eğit ---
model = LogisticRegression()
model.fit(X, y)

# --- 3️⃣ API uç noktası ---
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame([input_data])
    prediction = model.predict_proba(input_df)[0][1]  # probability of class 1
    return jsonify({'purchase_probability': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
