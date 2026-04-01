import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


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


# Build autoencoder model
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation="relu")(input_layer)
encoded = Dense(64, activation="relu")(encoded)

decoded = Dense(128, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


# Train model
autoencoder.fit(
    X_train,
    X_train,
    epochs=20,
    batch_size=128
)


# Reconstruction error
reconstructed = autoencoder.predict(X_test)

mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)

scores = (mse - mse.min()) / (mse.max() - mse.min())

threshold = np.percentile(scores, 95)

pred = (scores > threshold).astype(int)


# Evaluation metrics
roc = roc_auc_score(y_test, scores)
f1 = f1_score(y_test, pred)
pr = average_precision_score(y_test, scores)


print("GAN ROC-AUC:", roc)
print("GAN F1-score:", f1)
print("GAN PR-AUC:", pr)