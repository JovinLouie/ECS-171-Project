# streamlit front end for models
# joey suen

# packages
from xgboost import XGBClassifier
import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
    roc_auc_score, matthews_corrcoef, RocCurveDisplay
)
from log_reg_exported import engineer_features

# BACK END --------------------------------------------

set_random_seed = 171

@st.cache_data
def run_model(selection="XGBoost"):

    # load data, split into training and validation sets
    train_df = pd.read_csv("data/train_dataset.csv")
    val_df = pd.read_csv("data/val_dataset.csv")

    X_train = train_df.drop(columns=["Target_binary"])
    y_train = train_df["Target_binary"]
    X_val = val_df.drop(columns=["Target_binary"])
    y_val = val_df["Target_binary"]

    # option 1: xgboost model
    if selection=="XGBoost":
        scale = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=set_random_seed
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)
        y_prob = xgb_model.predict_proba(X_val)[:, 1]

        importances = xgb_model.feature_importances_
        order = np.argsort(importances)[::-1]
        top_k = min(20, len(importances))
        feat_names = X_train.columns

    # option 2: random forest
    elif selection=="Random Forest":
        rf_model = RandomForestClassifier(
            n_estimators=300,
            random_state=set_random_seed,
            class_weight="balanced"
        )
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_val)
        y_prob = rf_model.predict_proba(X_val)[:, 1]

        importances = rf_model.feature_importances_
        order = np.argsort(importances)[::-1]
        top_k = min(20, len(importances))
        feat_names = X_train.columns
    
    # option 3: logistic regression (+ engineered features)
    elif selection=="Logistic Regression":

        # engineered features
        X_train_eng = engineer_features(X_train)
        X_val_eng = engineer_features(X_val)
        
        # scaling data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_eng)
        X_val_scaled = scaler.transform(X_val_eng)  

        lr_model = LogisticRegression(
            C=0.1,
            solver='saga',
            class_weight='balanced',
            max_iter=1000,
            random_state=set_random_seed
        )
        lr_model.fit(X_train_scaled, y_train)
        y_pred = lr_model.predict(X_val_scaled)
        y_prob = lr_model.predict_proba(X_val_scaled)[:, 1]

        importances = np.abs(lr_model.coef_[0])
        order = np.argsort(importances)[::-1]
        top_k = min(20, len(importances))
        feat_names = X_train_eng.columns

    return y_pred.astype(int), y_val.to_numpy().astype(int), y_prob, importances, order, top_k, feat_names

# FRONT END --------------------------------------------

st.title("Student Dropout Predictor")

# model selection
selection = st.radio("Select a Model:", 
                         ["XGBoost", "Random Forest", "Logistic Regression"])

# evaluation on validation set
st.subheader(f"{selection} - Validation Set Metrics")

if st.button("Run Evaluation"):
    y_pred, y_val, y_prob, importances, order, top_k, feat_names = run_model(selection)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)

    col1, col2 = st.columns([3,2])
    with col1:
        col1a, col1b, col1c = st.columns(3)
        col1a.metric("Accuracy", f"{accuracy:.4f}")
        col1a.metric("Precision", f"{precision:.4f}")
        col1b.metric("Recall", f"{recall:.4f}")
        col1b.metric("F1", f"{f1:.4f}")
        col1c.metric("MCC", f"{mcc:.4f}")
        col1c.metric("ROC-AUC", f"{roc_auc:.4f}")
    
    with col2:
        st.write("Confusion Matrix")
        st.dataframe(pd.DataFrame(cm,
            index=["Actual: Not Dropout", "Actual: Dropout"],
            columns=["Predicted: Not Dropout", "Predicted: Dropout"]))
    

    st.write("")

    col7, col8 = st.columns([5,7])

    with col7:
        st.write("ROC Curve")
        fig_roc, ax = plt.subplots(figsize=(6,7))
        RocCurveDisplay.from_predictions(y_val, y_prob, ax=ax)
        ax.set_title(f"ROC Curve (AUC = {roc_auc:.4f})")
        st.pyplot(fig_roc)

    with col8:
        st.write("Feature Importances")
        fig = plt.figure(figsize=(8, 6))
        plt.barh(range(top_k), importances[order][:top_k][::-1])
        plt.yticks(range(top_k), [feat_names[i] for i in order[:top_k]][::-1])
        plt.xlabel("Importance")
        plt.title(f"Top {top_k} Feature Importances - {selection}")
        plt.tight_layout()
        st.pyplot(fig)
    