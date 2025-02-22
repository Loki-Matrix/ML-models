# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# read dataset
df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/TelecomCustomerChurn.csv')
df.head()
# define y and X
y = df['Churn']
X = df.drop(['customerID','Churn'],axis=1)
y.value_counts()
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X,y = ros.fit_resample(X,y)
y.value_counts()
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
X = oe.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2529)
# select mode
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))