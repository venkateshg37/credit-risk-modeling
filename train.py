import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Feature Engineering

# Example features (modify based on dataset columns)
# Assume columns: CreditAmount, Duration, Age

df["credit_utilization"] = df["CreditAmount"] / (df["CreditAmount"].max())

# Age binning 
df["age_group"] = pd.cut(df["Age"], bins=[18,30,50,100], labels=[0,1,2])

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Split
X = df.drop("Risk", axis=1)   # target column = Risk
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model trained successfully!")