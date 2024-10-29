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


clf = DecisionTreeClassifier(random_state=42, max_depth=5)  # Ví dụ: giới hạn độ sâu là 5
clf.fit(X_train, y_train.values.ravel())
y_pred = clf.predict(X_test)

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

# Lưu mô hình vào thư mục 'model' với tên 'decision_tree_model.joblib'
model_path = "models/decision_tree_model.joblib"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")