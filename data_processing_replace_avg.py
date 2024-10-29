import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, file_path, test_size=1/3.0, random_state=42):
        self.data = pd.read_csv(file_path)
        self.test_size = test_size
        self.random_state = random_state
        self.ms_col = []
        self.encode_data = self.data.copy()
        self.decode_ms = {}
        self.encode_ms = {}
        
        # Xử lý các giá trị missing
        self.find_missing_columns()
        self.calculate_missing_values()
        
        # Encode dữ liệu
        self.label_encode_data()
        self.handle_missing_encoded_values()

    def find_missing_columns(self):
        for col in self.data.columns:
            if "?" in np.unique(self.data[col]):
                self.ms_col.append(col)

    def calculate_missing_values(self):
        for col in self.ms_col:
            values, counts = np.unique(self.data[col], return_counts=True)
            self.decode_ms[col] = {value: count for value, count in zip(values, counts)}

    def label_encode_data(self):
        label_encoder = LabelEncoder()
        for col in self.encode_data.columns:
            self.encode_data[col] = label_encoder.fit_transform(self.encode_data[col])

    def handle_missing_encoded_values(self):
        for col in self.ms_col:
            values = np.unique(self.encode_data[col])
            avg = int(sum(values) / len(values))
            self.encode_data[col].replace(0, avg, inplace=True)

    def split_data(self):
        X = self.encode_data.iloc[:, 1:]
        y = self.encode_data.iloc[:, :1]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

# Sử dụng class
data_processor = Data("./data/agaricus-lepiota.data")
X_train, X_test, y_train, y_test = data_processor.split_data()

print("Length of X train:", len(X_train))
print("Length of X test:", len(X_test))
print("Length of y train:", len(y_train))
print("Length of y test:", len(y_test))
