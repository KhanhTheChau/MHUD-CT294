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


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

k_values = range(65, 81, 2)
accuracy_scores = []

for k in k_values:
    Model = KNeighborsClassifier(n_neighbors=k)
    Model.fit(X_train,y_train)
    y_pred = Model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    print(f"k = {k}, Accuracy KNN = {accuracy * 100:.2f}%")

k = k_values[np.argmax(accuracy_scores)]

print("Chon k = ",k)
Model_KNN = KNeighborsClassifier(n_neighbors=k)
Model_KNN.fit(X_train,y_train)

y_pred_KNN = Model_KNN.predict(X_test)

print(f"Accuracy KNN: {accuracy_scores[np.argmax(accuracy_scores)] * 100:.2f}%")
print(f"Precision KNN: {precision_score(y_test, y_pred_KNN) * 100:.2f}%")
print(f"Recall KNN: {recall_score(y_test, y_pred_KNN) * 100:.2f}%")
print(f"F1 KNN: {f1_score(y_test, y_pred_KNN) * 100:.2f}%")
print("Confusion matrix KNN:\n", confusion_matrix(y_test, y_pred_KNN, labels=np.unique(y_test)))
print("Classification Report KNN:\n", classification_report(y_test, y_pred_KNN))