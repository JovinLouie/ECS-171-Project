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


# MAPPINGS --------------------------------------------
marital_status_map = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated"
}
application_mode_map = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99 item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99 item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)"
}
course_map = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}
previous_qualification_map = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year - not completed",
    10: "11th year - not completed",
    12: "Other - 11th year",
    14: "10th year",
    15: "10th year - not completed",
    19: "Basic education 3rd cycle",
    38: "Basic education 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)"
}
nationality_map = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian"
}

# used for both father and mother features
qualification_map = { 
    1: "Secondary Education - 12th Year",
    2: "Higher Education - Bachelor's",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year - Not Completed",
    10: "11th Year - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year",
    13: "2nd year complementary high school",
    14: "10th Year",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle",
    20: "Complementary High School Course",
    22: "Technical-professional course",
    25: "Complementary High School - Not concluded",
    26: "7th Year",
    27: "2nd cycle general high school",
    29: "9th Year - Not Completed",
    30: "8th Year",
    31: "Administration and Commerce",
    33: "Supplementary Accounting and Administration",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without 4th year",
    37: "Basic education 1st cycle",
    38: "Basic Education 2nd Cycle",
    39: "Technological specialization",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies",
    42: "Professional higher technical",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)"
}
occupation_map = {
    0: "Student",
    1: "Directors and Executive Managers",
    2: "Intellectual and Scientific Specialists",
    3: "Intermediate Level Technicians",
    4: "Administrative staff",
    5: "Service and Sales Workers",
    6: "Agriculture/Fisheries Skilled Workers",
    7: "Industry and Construction Skilled Workers",
    8: "Machine Operators",
    9: "Unskilled Workers",
    10: "Armed Forces",
    90: "Other Situation",
    99: "Blank",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces personnel",
    112: "Administrative and commercial directors",
    114: "Hospitality directors",
    121: "Science and engineering specialists",
    122: "Health professionals",
    123: "Teachers",
    124: "Finance and accounting specialists",
    125: "ICT Specialists",
    131: "Science and engineering technicians",
    132: "Health technicians",
    134: "Legal/social/cultural technicians",
    135: "ICT technicians",
    141: "Office workers",
    143: "Accounting and registry operators",
    144: "Administrative support",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers",
    154: "Security personnel",
    161: "Market-oriented farmers",
    163: "Subsistence farmers/fishers",
    171: "Skilled construction workers",
    172: "Metalworking skilled workers",
    173: "Printing and craft workers",
    174: "Electrical skilled workers",
    175: "Food/clothing industry workers",
    181: "Plant and machine operators",
    182: "Assembly workers",
    183: "Drivers and mobile operators",
    191: "Cleaning workers",
    192: "Unskilled agriculture workers",
    193: "Unskilled industry workers",
    194: "Meal preparation assistants",
    195: "Street vendors"
}

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
    