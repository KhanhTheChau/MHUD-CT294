import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Đọc dữ liệu từ file
data = pd.read_csv("./data/agaricus-lepiota.data")

# Xác định các cột có giá trị missing (ký hiệu '?')
ms_col = []
for col in data.columns:
    if '?' in np.unique(data[col]):
        ms_col.append(col)
# Xử lý missing value bằng cách thay thế giá trị missing bằng giá trị phổ biến nhất (mode)
for col in ms_col:
    most_frequent_value = data[col].value_counts().idxmax()  # Tìm giá trị phổ biến nhất trong cột
    data[col] = data[col].replace('?', most_frequent_value)  # Thay thế '?' bằng giá trị phổ biến nhất

# Mã hóa các giá trị phân loại thành số
label_encoder = LabelEncoder()
for col in data.columns:
    data[col] = label_encoder.fit_transform(data[col])

# Tách đặc trưng (features) và nhãn (target)
X = data.iloc[:, 1:]  # Tất cả các cột trừ cột đầu tiên (nhãn)
y = data.iloc[:, 0]   # Cột đầu tiên là nhãn

# Loại bỏ các đặc trưng có tầm quan trọng bằng 0 dựa trên kết quả trước đó
# Giả sửa cột p.2 có importance bằng 0
X = X.drop(columns=['p.2'])  # Bạn có thể thêm nhiều cột nếu cần

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Sử dụng GridSearchCV để tìm tham số tối ưu cho Decision Tree
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [10, 15, 20],
    'min_samples_leaf': [5, 10, 15]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất từ GridSearchCV
dt_clf_best = grid_search.best_estimator_

# Dự đoán trên tập kiểm thử
y_pred_dt = dt_clf_best.predict(X_test)

# Đánh giá mô hình Decision Tree tốt nhất
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

# In các chỉ số đánh giá
print(f"Decision Tree - Accuracy: {accuracy_dt:.4f}, Precision: {precision_dt:.4f}, Recall: {recall_dt:.4f}, F1 Score: {f1_dt:.4f}")

# Sử dụng Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(dt_clf_best, X, y, cv=skf)

print(f"Stratified K-Fold cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

# Hiển thị độ quan trọng của các đặc trưng (feature importance)
importances = dt_clf_best.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Nếu muốn lưu mô hình đã huấn luyện để sử dụng sau:
import joblib
joblib.dump(dt_clf_best, 'decision_tree_best_model.pkl')
