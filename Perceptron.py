from data_processing_replace_avg import Data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Khởi tạo và xử lý dữ liệu
data_processor = Data("./data/agaricus-lepiota.data")
X_train, X_test, y_train, y_test = data_processor.split_data()

print("Length of X train:", len(X_train))
print("Length of X test:", len(X_test))
print("Length of y train:", len(y_train))
print("Length of y test:", len(y_test))


from sklearn.linear_model import Perceptron
import pandas as pd
import pickle
import os

# Khởi tạo mô hình Perceptron
perceptron_model = Perceptron(shuffle=True, random_state=0)

# Huấn luyện mô hình với dữ liệu huấn luyện
perceptron_model.fit(X_train, y_train.values.ravel())

# Dự đoán với dữ liệu kiểm tra
y_pred = perceptron_model.predict(X_test)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)

# Tạo báo cáo phân loại
report = classification_report(y_test, y_pred, output_dict=True)

# Chuyển đổi báo cáo thành DataFrame để dễ đọc
report_df = pd.DataFrame(report).transpose()

# In ra độ chính xác và báo cáo phân loại
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report_df)

print(confusion_matrix(y_pred, y_test))


# Lưu mô hình vào tệp 'Perceptron_Model.sav' trong thư mục 'model'
os.makedirs("models", exist_ok=True)
model_path = "models/Perceptron_Model.sav"
pickle.dump(perceptron_model, open(model_path, "wb"))
print(f"Model saved to {model_path}")
