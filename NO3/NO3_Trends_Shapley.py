# NO3 Trends - Shapley Values

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import xgboost as xgb
import shap

# Load Dataset
# DATA = readRDS(paste0(strap_dir,'Data/Models/RF_bi_model_All_DATA_all_vars_','Trends_Conc_PWS_GW_05to20', '.rds'))
strap_dir = 'C:/Users/MPennino/OneDrive - Environmental Protection Agency (EPA)/Projects/StRAPs/StRAP4/SSWR.405.1_NO3_Trend_Causes/Data/Models/'
FILE_NAME = 'RF_bi_model_All_DATA_all_vars_Trends_Conc_PWS_GW_05to20.csv'
df = pd.read_csv(strap_dir+FILE_NAME)
df.head(3)

# Split Dataset 
X = df[:, :-1]
y = df[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluation of the Model
ccuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
            xticklabels=iris.target_names, yticklabels=iris.target_names)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Shapley Analysis
# 1. Load a standard dataset and split it
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
rf_model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# 3. Initialize the SHAP Explainer
# The explainer automatically selects an appropriate algorithm (e.g., TreeSHAP for tree models)
#explainer = shap.Explainer(model)
explainer = shap.TreeExplainer(rf_model)

# 4. Calculate SHAP values for the test set
shap_values = explainer(X_test)

# 5. Visualize the results
# See the visualization section for different plot types