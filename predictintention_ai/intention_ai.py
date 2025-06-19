from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Yeni cart.csv dosyasını oku
df = pd.read_csv('cart.csv')

# Sayısal feature’ları seçiyoruz
features = ['price', 'session_duration', 'pages_viewed', 'returning_user', 'discount_shown']

X = df[features]
y = df['purchased']

# Modeli eğit
model = LogisticRegression(max_iter=1000).fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Gelen veride feature’ları alıyoruz, yoksa hata verir
    try:
        input_data = [[
            data['price'],
            data['session_duration'],
            data['pages_viewed'],
            data['returning_user'],
            data['discount_shown']
        ]]
    except KeyError as e:
        return jsonify({'error': f"Eksik alan: {str(e)}"}), 400

    # Tahmin olasılığı hesapla
    prob = model.predict_proba(input_data)[0][1]

    return jsonify({"purchase_probability": prob})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)

