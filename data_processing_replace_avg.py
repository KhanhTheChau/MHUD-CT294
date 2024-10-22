# %%
import pandas as pd
import numpy as np

data = pd.read_csv("./data/agaricus-lepiota.data")
data

# %%
for col in data.columns:
    print(f"{col}: {np.unique(data[col])}")

# %%
ms_col = []
for col in data.columns:
    value_col = np.unique(data[[col]])
    for value in value_col:
        if value == "?":
            ms_col.append(col)
            
ms_col

# %%
encode_data = data

decode_ms = {}

for col in ms_col:
    values, counts = np.unique(encode_data[col], return_counts=True)
    decode_ms[col] = {}
    for value, count in zip(values, counts):
        decode_ms[col][value] = count 


# %%
# Convert data to numeric value
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for col in encode_data.columns:
    encode_data[col] = label_encoder.fit_transform(encode_data[col])

for col in encode_data.columns:
    print(f"{col}: {np.unique(encode_data[col])}, type: {encode_data[col].dtype}")
    

# %%
encode_ms = {}

for col in ms_col:
    values, counts = np.unique(encode_data[col], return_counts=True)
    encode_ms[col] = {}
    for value, count in zip(values, counts):
        encode_ms[col][value] = count 


# %%
print(encode_ms)
print(decode_ms)

# %%
# Match encode_ms and decode_ms we will see LableEncoder convert missing values to different values
for col in ms_col:
    values = np.unique(encode_data[col])

    avg = int(sum(values)/(len(values))) 
    print(f"{col}: avg = {avg}")
    encode_data[col].replace(0, avg, inplace=True)

ms_col

# %%
for col in encode_data.columns:
    print(f"{col}: {np.unique(encode_data[col])}, type: {encode_data[col].dtype}")

# %%
from sklearn.model_selection import train_test_split

X = data.iloc[:,1:]
y = data.iloc[:,:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=42)
print("Length of X train: ", len(X_train))
print("Length of X test: ", len(X_test))
print("Length of y train: ", len(y_train))
print("Length of y test: ", len(y_test))


