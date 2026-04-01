import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Load dataset
transactions = pd.read_csv("train_transaction.csv")
identity = pd.read_csv("train_identity.csv")


# Merge datasets
data = transactions.merge(identity, on="TransactionID", how="left")


# Remove columns with too many missing values
missing_ratio = data.isnull().mean()
data = data.loc[:, missing_ratio < 0.8]


# Remove constant columns
constant_cols = data.columns[data.nunique(dropna=False) <= 1]
data = data.drop(columns=constant_cols)


# Separate target and features
y = data["isFraud"]
X = data.drop(["isFraud", "TransactionID"], axis=1)


# Identify column types
num_cols = X.select_dtypes(exclude=["object"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns


# Fill missing values
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("Unknown")


# Encode categorical variables
le = LabelEncoder()

for c in cat_cols:
    X[c] = le.fit_transform(X[c].astype(str))


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# CNN style neural network
model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)


# Train model
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128
)


# Predictions
pred_prob = model.predict(X_test)
pred = (pred_prob > 0.5).astype(int)


# Evaluation metrics
roc = roc_auc_score(y_test, pred_prob)
f1 = f1_score(y_test, pred)
pr = average_precision_score(y_test, pred_prob)


print("ROC-AUC:", roc)
print("F1-score:", f1)
print("PR-AUC:", pr)