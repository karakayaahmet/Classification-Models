import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Veri Seti Hikayesi ve Problem: Şeker Hastalığı Tahmini

# YAPAY SİNİR AĞLARI (ÇOK KATMANLI ALGILAYICILAR)

df = pd.read_csv("diabetes.csv")
print(df.head())

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Model & Tahmin

mlpc_model = MLPClassifier().fit(X_train, y_train)
print(mlpc_model.coefs_)
print(mlpc_model.intercepts_)

y_pred = mlpc_model.predict(X_test)
accuracy_score(y_test, y_pred)


