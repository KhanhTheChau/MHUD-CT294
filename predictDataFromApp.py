import numpy as np
import pickle
import pandas as pd

# Load the model
with open('./models/SVM_Model.sav', 'rb') as f:
    model = pickle.load(f)

class PredictDataFromFeatures:    
    def __init__(self, data):
        self.data = data
        self.transformed_data = []
        self.encode_list = [[0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0],
            [0, 1, 2, 3],
            [0, 1, 2],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5, 6]]  
        self.decode_list = [['b' 'c' 'f' 'k' 's' 'x'],
            ['f' 'g' 's' 'y'],
            ['b' 'c' 'e' 'g' 'n' 'p' 'r' 'u' 'w' 'y'],
            ['f' 't'],
            ['a' 'c' 'f' 'l' 'm' 'n' 'p' 's' 'y'],
            ['a' 'f'],
            ['c' 'w'],
            ['b' 'n'],
            ['b' 'e' 'g' 'h' 'k' 'n' 'o' 'p' 'r' 'u' 'w' 'y'],
            ['e' 't'],
            ['?' 'b' 'c' 'e' 'r'],
            ['f' 'k' 's' 'y'],
            ['f' 'k' 's' 'y'],
            ['b' 'c' 'e' 'g' 'n' 'o' 'p' 'w' 'y'],
            ['b' 'c' 'e' 'g' 'n' 'o' 'p' 'w' 'y'],
            ['p'],
            ['n' 'o' 'w' 'y'],
            ['n' 'o' 't'],
            ['e' 'f' 'l' 'n' 'p'],
            ['b' 'h' 'k' 'n' 'o' 'r' 'u' 'w' 'y'],
            ['a' 'c' 'n' 's' 'v' 'y'],
            ['d' 'g' 'l' 'm' 'p' 'u' 'w']]  
        self.combine_ed = [dict(zip(enc, dec)) for enc, dec in zip(self.encode_list, self.decode_list)]

    def Transform(self):
        for feature, encoding_dict in zip(self.data, self.combine_ed):
            self.transformed_data.append(encoding_dict.get(feature, None))

    def ReplaceMissingByAVG(self):
        for i in range(len(self.data)):
            if self.transformed_data[i] is None:  
                avg_value = sum(self.combine_ed[i]) // len(self.combine_ed[i])  
                self.transformed_data[i] = avg_value

    def Predict(self):
        self.Transform()
        self.ReplaceMissingByAVG()
        result = model.predict([self.transformed_data])  
        return "edible" if result == 0 else "poisonous"

class PredictDataFromCSV:
    def __init__(self, data):
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])  
        self.data = data
        self.encode_data = data.copy()
        self.missing_col = []

    def ReplaceNullByMissing(self):
        self.data.fillna("?", inplace=True)

    def FindMissingValue(self):
        for col in self.data.columns:
            if "?" in self.data[col].values:
                self.missing_col.append(col)

    def LabelData(self):
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()
        for col in self.data.columns:
            self.encode_data[col] = label_encoder.fit_transform(self.data[col])

    def ReplaceMissingByAVG(self):
        for col in self.missing_col:
            unique_vals = np.unique(self.encode_data[col])
            avg = int(unique_vals.mean())
            self.encode_data[col].replace(0, avg, inplace=True)

    def Predict(self):
        self.ReplaceNullByMissing()
        self.FindMissingValue()
        self.LabelData()
        self.ReplaceMissingByAVG()

        label = ["edible", "poisonous"]
        predict = model.predict(self.encode_data.values)
        predict = [label[i] for i in predict]
        self.data["Predict"] = predict
        self.data.to_csv("result.csv", index=True)  
        return "Prediction completed and saved to result.csv."
