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


from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import pickle
import os

# Khởi tạo và huấn luyện mô hình SVM với kernel "linear"
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train, y_train.values.ravel())

# Dự đoán với dữ liệu kiểm tra
y_pred = svm_model.predict(X_test)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Chuyển đổi report thành DataFrame và lọc ra các lớp riêng biệt (bỏ accuracy, macro avg, weighted avg)
report_df = pd.DataFrame(report).transpose().loc[["0", "1"]]

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report_df)

print(confusion_matrix(y_pred, y_test))

# import matplotlib.pyplot as plt
# import seaborn as sns
# # Vẽ confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
#             xticklabels=["Class 0", "Class 1"], 
#             yticklabels=["Class 0", "Class 1"])
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.title("Confusion Matrix")
# plt.show()

# Lưu mô hình vào tệp 'SVM_Model.sav' trong thư mục 'model'
os.makedirs("models", exist_ok=True)
model_path = "models/SVM_Model.sav"
pickle.dump(svm_model, open(model_path, "wb"))
print(f"Model saved to {model_path}")
