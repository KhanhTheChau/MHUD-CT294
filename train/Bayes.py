import numpy as np
from data_processing_replace_avg import Data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Khởi tạo và xử lý dữ liệu
data_processor = Data("./data/agaricus-lepiota.data")
X_train, X_test, y_train, y_test = data_processor.split_data()

print("Length of X train:", len(X_train))
print("Length of X test:", len(X_test))
print("Length of y train:", len(y_train))
print("Length of y test:", len(y_test))

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

model_BN=GaussianNB()
model_BN.fit(X_train,y_train)

y_pred_Bayes = model_BN.predict(X_test)

print(f"Accuracy Bayes: {accuracy_score(y_test,y_pred_Bayes) * 100:.2f}%")
print(f"Precision KNN: {precision_score(y_test, y_pred_Bayes) * 100:.2f}%")
print(f"Recall KNN: {recall_score(y_test, y_pred_Bayes) * 100:.2f}%")
print(f"F1 KNN: {f1_score(y_test, y_pred_Bayes) * 100:.2f}%")
print("Confusion Matrix Bayes: \n",confusion_matrix(y_test, y_pred_Bayes, labels=np.unique(y_test)))
print("Classification Report Bayes:\n", classification_report(y_test, y_pred_Bayes))