import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.neighbors import NearestNeighbors


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


# Convert to tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)


# Build graph using KNN
nbrs = NearestNeighbors(n_neighbors=5)
nbrs.fit(X_train)

distances, indices = nbrs.kneighbors(X_train)

edge_index = []

for i in range(len(indices)):
    for j in indices[i]:
        edge_index.append([i, j])

edge_index = torch.tensor(edge_index).t().contiguous()


data_graph = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)


# Define GNN model
class GNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(X_tensor.shape[1], 64)
        self.conv2 = GCNConv(64, 32)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.linear(x)

        return x


model = GNN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()


# Train model
for epoch in range(50):

    optimizer.zero_grad()

    out = model(data_graph).squeeze()

    loss = loss_fn(out, data_graph.y)

    loss.backward()
    optimizer.step()


# Evaluation
model.eval()

with torch.no_grad():

    logits = model(data_graph).squeeze()
    probs = torch.sigmoid(logits).numpy()


pred = (probs > 0.5).astype(int)

roc = roc_auc_score(y_train, probs)
f1 = f1_score(y_train, pred)
pr = average_precision_score(y_train, probs)


print("GNN ROC-AUC:", roc)
print("GNN F1-score:", f1)
print("GNN PR-AUC:", pr)