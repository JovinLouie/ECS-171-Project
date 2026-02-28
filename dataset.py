# Jovin Louie
# ECS 171 Project: Student Dropout Prediction

set_random_seed = 171

import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dropout_data.csv", sep=";")

# Create binary target
df["Target_binary"] = df["Target"].apply(
    lambda x: 1 if x == "Dropout" else 0
)

# Selected features
selected_features = [
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (evaluations)",
    "Tuition fees up to date",
    "Curricular units 1st sem (evaluations)",
    "Admission grade",
    "Course",
    "Age at enrollment",
    "Previous qualification (grade)",
    "Father's occupation",
    "Application mode",
    "Mother's occupation",
    "Curricular units 2nd sem (enrolled)",
    "GDP",
    "Curricular units 1st sem (enrolled)",
    "Mother's qualification",
    "Father's qualification",
    "Scholarship holder"
]

# Define X and y
X = df[selected_features]
y = df["Target_binary"]

# 60-15–25 stratified split (train-val-test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y,
    random_state=set_random_seed
)

# Split temp into 15% validation and 25% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.625,  # 25 / 40
    stratify=y_temp,
    random_state=set_random_seed
)

# Combine features + target for export
train_df = X_train.copy()
train_df["Target_binary"] = y_train

test_df = X_test.copy()
test_df["Target_binary"] = y_test

val_df = X_val.copy()
val_df["Target_binary"] = y_val

# Export datasets
train_df.to_csv("data/train_dataset.csv", index=False)
test_df.to_csv("data/test_dataset.csv", index=False)
val_df.to_csv("data/val_dataset.csv", index=False)

print("Training, validation, and test datasets exported successfully.")