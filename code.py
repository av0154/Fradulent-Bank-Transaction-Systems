
import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')
# Read Raw Dataset

!ls "/content/drive/MyDrive/Dataset_for_ds/2.csv/fraud_detection_bank_dataset (1).csv"
!ls "/content/drive/My Drive/Dataset_for_ds/2.csv/PS_20174392719_1491204439457_log.csv"

import pandas as pd
import numpy as np

data1 = pd.read_csv("/content/drive/MyDrive/Dataset_for_ds/2.csv/fraud_detection_bank_dataset (1).csv")
data2 = pd.read_csv("/content/drive/My Drive/Dataset_for_ds/2.csv/PS_20174392719_1491204439457_log.csv")

merged_df = pd.merge(data1, data2, on=['amount'], how= 'inner')

print(data1)

print(data2)

merged_df = merged_df.drop('Unnamed: 0', axis = 1)
merged_df.head

print(merged_df)

def custom_align(merged_df):
    return 'text-align: center'

merged_df

print(data1.columns)
print(data2.columns)

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming merged_df is your DataFrame containing data

fig, axes = plt.subplots(7, 16, figsize=(30, 15))

# Flatten the axes array to easily iterate over the subplots
axes = axes.flatten()

for col, ax in enumerate(axes):
    if col < len(merged_df.columns):  # Ensure we don't exceed the number of columns
        column_name = merged_df.columns[col]
        sns.kdeplot(data=merged_df, x=column_name, fill=True, ax=ax, warn_singular=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(column_name, loc='center', weight='bold', fontsize=10)

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

print("col_14 counts : ",(merged_df['col_14'] > merged_df['col_14'].mean()).sum())
print("col_15 counts : ",(merged_df['col_15'] > merged_df['col_15'].mean()).sum())
print("col_96 counts : ",(merged_df['col_96'] > merged_df['col_96'].mean()).sum())
print("col_97 counts : ",(merged_df['col_97'] > merged_df['col_97'].mean()).sum())
print("col_98 counts : ",(merged_df['col_98'] > merged_df['col_98'].mean()).sum())

numeric_df = merged_df.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_df.drop(['targets'], axis=1).corr()
plt.figure(figsize=(25, 18))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

numeric_df_var = numeric_df.loc[:,numeric_df.nunique()>1] #This way we remove the columns with the same value in every row

correlation_matrix = numeric_df_var.drop(['targets'], axis =1).corr().abs()

# Select upper triangular of correlation matrix (excluding diagonal)
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find index of columns with correlation above threshold
columns_to_drop = [column for column in upper.columns if any(upper[column] >= 0.8)]

# Drop the highly correlated columns
numeric_df_filtered = numeric_df_var.drop(columns_to_drop, axis=1)

correlation_matrix = numeric_df_filtered.drop(['targets'], axis =1).corr().abs()
plt.figure(figsize=(25, 18))
sns.heatmap(correlation_matrix, annot=False, cmap='Blues', square=True)
plt.title('Correlation Matrix')
plt.show()

numeric_df_filtered.isnull().sum()

"""Scaling

"""

numeric_df_filtered.targets.value_counts()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df_filtered.drop('targets', axis=1))
y = numeric_df_filtered['targets']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train.shape

"""Oversample"""

#Import and use SMOTE to oversample
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
x_train_resampled.shape

#The training set is currently balanced, and the test set has not information about these objervation
y_train_resampled.value_counts()

"""ANN Model"""

#Create the model

from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

i = Input(shape =(76,) )
x = Dropout(0.2)(i)
x = Dense(64, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation = 'relu')(x)
x = Dense(1, activation = 'sigmoid')(x)
model = Model(i,x)

model.summary()

from tensorflow.keras.metrics import Recall

model.compile(optimizer='adam', loss='binary_crossentropy', metrics= Recall())

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model without early stopping
r = model.fit(x_train_resampled, y_train_resampled, validation_data=(X_test, y_test), batch_size=32, epochs=100)

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()


plt.plot(r.history['recall'], label = 'recall')
plt.plot(r.history['val_recall'], label = 'val_recall')
plt.legend()
plt.show();

"""Results

"""

from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score, roc_curve
from sklearn import metrics
y_pred = model.predict(X_test)
pred_class =np.where(y_pred < 0.5,0,1)
cm = confusion_matrix(y_test.astype(int), pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Non-Fraud','Fraud'])
cm_display.plot()
plt.show()

f1_score(y_test.astype(int), pred_class)

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(y_test,y_pred)

pred_class =np.where(y_pred < 0.8,0,1) #changing the threshold to 0.8
cm = confusion_matrix(y_test.astype(int), pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Non-Fraud','Fraud'])
cm_display.plot()
plt.show()

f1_score(y_test.astype(int), pred_class)

from sklearn.metrics import recall_score
recall_score(y_test.astype(int), pred_class)

pred_class =np.where(y_pred < 0.95,0,1) #changing the threshold to 0.8
cm = confusion_matrix(y_test.astype(int), pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Non-Fraud','Fraud'])
cm_display.plot()
plt.show()
