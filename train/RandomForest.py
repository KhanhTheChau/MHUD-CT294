from data_processing_replace_avg import Data

# Khởi tạo và xử lý dữ liệu
data_processor = Data("./data/agaricus-lepiota.data")
X_train, X_test, y_train, y_test = data_processor.split_data()

print("Length of X train:", len(X_train))
print("Length of X test:", len(X_test))
print("Length of y train:", len(y_train))
print("Length of y test:", len(y_test))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Khởi tạo mô hình Random Forest
clf = RandomForestClassifier(random_state=42, max_depth=3)

# Huấn luyện mô hình với dữ liệu huấn luyện
clf.fit(X_train, y_train.values.ravel())  # Sử dụng .values.ravel() để chuyển y_train thành mảng một chiều

# Dự đoán với dữ liệu kiểm tra
y_pred = clf.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

print(confusion_matrix(y_pred, y_test))

import joblib
import os

# Tạo thư mục 'model' nếu chưa tồn tại
os.makedirs("models", exist_ok=True)

# Lưu mô hình vào thư mục 'model' với tên 'random_forest_model.joblib'
model_path = "models/random_forest_model.joblib"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")
