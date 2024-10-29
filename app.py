from flask import Flask, request, render_template, redirect, url_for
import pickle
import pandas as pd

app = Flask(__name__)

# Tải mô hình
with open('./models/decision_tree_model.joblib', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nếu người dùng nhập trực tiếp
    if request.form.get("form-type") == "manual":
        features = [request.form[f] for f in [
            'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]]
        data = pd.DataFrame([features])

    # Nếu người dùng tải lên file CSV
    elif request.form.get("form-type") == "csv":
        file = request.files['file']
        data = pd.read_csv(file)

    # Dự đoán
    # predictions = model.predict(data)
    result = "ăn được"
    # result = ["Ăn được" if pred == 'e' else "Độc" for pred in predictions]
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
