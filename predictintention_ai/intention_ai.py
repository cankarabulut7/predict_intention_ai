from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

DATA_FILE = 'cart.csv'
data = pd.read_csv(DATA_FILE)

X = data[['price', 'session_duration', 'pages_viewed', 'returning_user', 'discount_shown']]
y = data['purchase_made']  # 0 veya 1 !!!

model = LogisticRegression()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame([input_data])
    prediction = model.predict_proba(input_df)[0][1]  # olasılık
    return jsonify({'purchase_probability': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
