from flask import Flask, request, render_template, redirect, url_for, send_file
from predictDataFromApp import *
from FindNameMushroom import *
from DownloadImageGoogle import *

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
        
        search_dict = {
            "cap-shape": input[0],
            "stalk-surface-above-ring": input[11],
            "veil-color": input[17],
            "habitat": input[21],
            "population": input[20]
        }
        
        prompt = "Tìm tên theo các thuộc tính sau: "
        for key in prompt_dict.keys():
            values_prompt_dict = prompt_dict[key]
            search_dict[str(key)] = values_prompt_dict[str(search_dict[str(key)])]
        
        for key, value in search_dict.items():
            prompt += f"{key}: {value}\n"
        prompt += f"Chỉ tìm tên không tìm cái khác"
        
        mushroom_name = FindName(prompt)
        
        data = PredictDataFromFeatures(input)
        result = data.Predict()
     
        download_images(mushroom_name, num_images=1)
     
    # Nếu người dùng tải lên file CSV
    elif request.form.get("form-type") == "csv":
        input = request.files['file']
        data = pd.read_csv(input)
        data = PredictDataFromCSV(data)
        result = data.Predict()
            
        return send_file("./result.csv", as_attachment=True)

   
    return render_template('index.html', result=result, mushroom_name=mushroom_name)

if __name__ == '__main__':
    app.run(debug=True)
