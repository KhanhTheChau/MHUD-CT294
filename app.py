from flask import Flask, request, render_template, redirect, url_for, send_file
from predictDataFromApp import *

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nếu người dùng nhập trực tiếp
    if request.form.get("form-type") == "manual":
        input = [request.form[f] for f in [
            'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]]
        
        data = PredictDataFromFeatures(input)
        result = data.Predict()
     

    # Nếu người dùng tải lên file CSV
    elif request.form.get("form-type") == "csv":
        input = request.files['file']
        data = pd.read_csv(input)
        data = PredictDataFromCSV(data)
        result = data.Predict()
            
        return send_file("./result.csv", as_attachment=True)

   
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
