#!/usr/bin/env python
# coding: utf-8

# This has been adapted from a jupyter notebook so it may need adjustments to work properly and may even work better if
# pasted back into a jupyter notebook

# In[3]:


# !pip install catboost
# !pip install shap
from catboost import *
from PIL import Image, ImageDraw, ImageEnhance

import pickle
import nltk
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import time
import numpy as np
from catboost import CatBoost, Pool
from catboost.utils import get_confusion_matrix
import shap
from shap import Explainer
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import os


# In[12]:


df = pd.read_csv(r"C:\Users\emwil\cs_projects\BoldDetection\data.csv", encoding='utf-8-sig')
print(df)


# In[6]:


for i in df.columns:
    print(i, df[i].dtype)


# In[7]:


# create and train model
model = CatBoostClassifier(custom_metric=['Logloss', 'AUC:hints=skip_train~false'])
feature_names = [i for i in df.columns if (df[i].dtype in [np.int64, np.float64])]
important_features = ['rel_height', 'rel_width', 'rel_size', 'col_aprox', 'Line Location', 'rel_left', 'rel_right', 'rel_top', 'rel_bottom']
X = df[important_features]
print(important_features)
print(X.head())
end = 146326
X_train = X[:end]
X_test = X[end+1:]
y_train = df['Bolded'][:end]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[8]:


# get accuracy of prediction
y_prediction = [ True if x == "True" else False for x in y_pred]
actual_bold = df['Bolded'][end+1:]
print(len(y_prediction), len(actual_bold))
print('The accuracy of the catboost model is :\t',metrics.accuracy_score(y_prediction, actual_bold))


# In[9]:


cm = confusion_matrix(actual_bold, y_prediction)
print(cm)
print("True Negative: " + str(cm[0, 0]))
print("True Positive: " + str(cm[1, 1]))
print("False Negative: " + str(cm[1, 0]))
print("False Positive: " + str(cm[0, 1]))
print("recall: " + str(metrics.recall_score(actual_bold, y_prediction)))
print("precision: " + str(metrics.precision_score(actual_bold, y_prediction)))
print("accuracy score: " + str(accuracy_score(actual_bold, y_prediction)))


# In[10]:


# previous results before normalizing coordinates:
# [[144813    152]
#  [   903    835]]
# True Negative: 144813
# True Positive: 835
# False Negative: 903
# False Positive: 152
# recall: 0.48043728423475257
# precision: 0.8459979736575481
# accuracy score: 0.9928085996878047

# Results After Normalized Coordinates
# [[144789    176]
#  [   867    871]]
# True Negative: 144789
# True Positive: 871
# False Negative: 867
# False Positive: 176
# recall: 0.501150747986191
# precision: 0.8319006685768864
# accuracy score: 0.9928903976060476


# In[11]:


shap.initjs()

explainer = Explainer(model)
shap_values = explainer(X_train)


# In[12]:


shap.plots.bar(shap_values, show=True, max_display=16)
shap.plots.beeswarm(shap_values, max_display=16)


# In[13]:


for i in range(len(important_features)):
  shap.dependence_plot(i, shap_values.values, X_train)


# In[14]:


model_flip = CatBoostClassifier(custom_metric=['Logloss', 'AUC:hints=skip_train~false'])
# feature_names = [i for i in df.columns if (df[i].dtype in [np.int64, np.float64])]
important_features = ['rel_height', 'rel_width', 'rel_size', 'col_aprox', 'Line Location', 'rel_left', 'rel_right', 'rel_top', 'rel_bottom']
X = df[important_features]
print(important_features)
print(X.head())
end = 146326
X_train_flip = X[end+1:]
X_test_flip = X[:end]
y_train_flip = df['Bolded'][end+1:]
model_flip.fit(X_train_flip, y_train_flip)
y_pred_flip = model_flip.predict(X_test_flip)


# In[15]:


# get accuracy of flipped prediction
y_prediction_flip = [ True if x == "True" else False for x in y_pred_flip]
actual_bold_flip = df['Bolded'][:end]
print(len(y_prediction_flip), len(actual_bold_flip))
print('The accuracy of the flipped catboost model is :\t',metrics.accuracy_score(y_prediction_flip, actual_bold_flip))


# In[16]:


cm = confusion_matrix(actual_bold_flip, y_prediction_flip)
print("Accuracy and Confusion Matrix for Flipped Data")
print(cm)
print("True Negative: " + str(cm[0, 0]))
print("True Positive: " + str(cm[1, 1]))
print("False Negative: " + str(cm[1, 0]))
print("False Positive: " + str(cm[0, 1]))
print("recall: " + str(metrics.recall_score(actual_bold_flip, y_prediction_flip)))
print("precision: " + str(metrics.precision_score(actual_bold_flip, y_prediction_flip)))
print("accuracy score: " + str(accuracy_score(actual_bold_flip, y_prediction_flip)))


# In[17]:


# Accuracy and Confusion Matrix for Flipped Data
# [[145234      0]
#  [    10   1082]]
# True Negative: 145234
# True Positive: 1082
# False Negative: 10
# False Positive: 0
# recall: 0.9908424908424909
# precision: 1.0
# accuracy score: 0.999931659445348

# Accuracy and Confusion Matrix for Flipped Data
# [[145139     95]
#  [   408    684]]
# True Negative: 145139
# True Positive: 684
# False Negative: 408
# False Positive: 95
# recall: 0.6263736263736264
# precision: 0.8780487804878049
# accuracy score: 0.9965624701010073


# In[18]:


shap.initjs()

explainer = Explainer(model_flip)
shap_values_flip = explainer(X_train_flip)


# In[19]:


shap.plots.bar(shap_values_flip, show=True, max_display=16)
shap.plots.beeswarm(shap_values_flip, max_display=16)


# In[20]:


for i in range(len(important_features)):
  shap.dependence_plot(i, shap_values_flip.values, X_train_flip, interaction_index=None)


# In[ ]:


# shap.dependence_plot(2, shap_values_flip.values, X_train_flip, xmax=1000, interaction_index=None, alpha=.1)
# shap.dependence_plot(2, shap_values.values, X_train, xmax=1000, interaction_index=None, alpha=.1)


# In[ ]:


# shap.dependence_plot(0, shap_values_flip.values, X_train_flip, xmax=30, interaction_index=None, alpha=.1)
# shap.dependence_plot(0, shap_values.values, X_train, xmax=30, interaction_index=None, alpha=.1)


# In[44]:


a = df[:226185]
b = df[232726:283405]
c = [a, b]
two_cols = pd.concat(c)
two_cols.to_csv(r"C:\Users\emwil\Downloads\two_cols.csv", encoding='utf-8-sig', index=False)
model = CatBoostClassifier(custom_metric=['Logloss', 'AUC:hints=skip_train~false'])
important_features = ['rel_height', 'rel_width', 'rel_size', 'col_aprox', 'Line Location', 'rel_left', 'rel_bottom', 'line width']
X = two_cols[important_features]
end = 133296
X_train = X[:end]
X_test = X[end+1:]
y_train = two_cols['Bolded'][:end]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#                       'line width', 'line height', 'charachters per line']
# new_imp = ['Book Name', 'Page Number', 'Coordinates', 'rel_height', 'rel_width', 'rel_size', 'col_aprox', 'Line Location', 'rel_left', 'rel_bottom', 'line width']
# print(two_cols[important_features])
# print(important_features)

# print(X_train.head())
# print(y_train.head())


# In[45]:


# get accuracy of prediction
y_prediction = [ True if x == "True" else False for x in y_pred]
actual_bold = two_cols['Bolded'][end+1:]
print(len(y_prediction), len(actual_bold))
print('The accuracy of the catboost model is :\t',metrics.accuracy_score(y_prediction, actual_bold))
cm = confusion_matrix(actual_bold, y_prediction)
print(cm)
print("True Negative: " + str(cm[0, 0]))
print("True Positive: " + str(cm[1, 1]))
print("Prediction: Not. Actual: Bold: " + str(cm[1, 0]))
print("Prediction: Bold. Actual: Not: " + str(cm[0, 1]))
print("recall: " + str(metrics.recall_score(actual_bold, y_prediction)))
print("precision: " + str(metrics.precision_score(actual_bold, y_prediction)))
print("accuracy score: " + str(accuracy_score(actual_bold, y_prediction)))


# In[70]:


X_test.head()
td = two_cols[end+1:]
print(type(td['Book Name'].tolist()), type(y_prediction))
new_data = [td['Book Name'].tolist(), td['Page Number'].tolist(), td['Coordinates'].tolist(), y_prediction]
print("made data")
pred_df = pd.DataFrame(new_data).transpose()
print(pred_df)
pred_df.columns= ['Book', 'Page', 'Coors', 'Predict']
pred_df.to_csv(r"C:\Users\emwil\Downloads\pred_df.csv", index=False)


# In[64]:


pred_df


# In[37]:


shap.initjs()

explainer = Explainer(model)
shap_values = explainer(X_train)


# In[38]:


shap.plots.bar(shap_values, show=True, max_display=16)
shap.plots.beeswarm(shap_values, max_display=16)
for i in range(len(important_features)):
  shap.dependence_plot(i, shap_values.values, X_train, interaction_index=None)


# In[7]:


# Now Flipping it
model_flip = CatBoostClassifier(custom_metric=['Logloss', 'AUC:hints=skip_train~false'])
X_train_flip = X[end+1:]
X_test_flip = X[:end]
y_train_flip = two_cols['Bolded'][end+1:]
model_flip.fit(X_train_flip, y_train_flip)
y_pred_flip = model_flip.predict(X_test_flip)


# In[15]:


# get accuracy of flipped prediction
y_prediction_flip = [ True if x == "True" else False for x in y_pred_flip]
actual_bold_flip = two_cols['Bolded'][:end]
print(len(y_prediction_flip), len(actual_bold_flip))
print('The accuracy of the flipped catboost model is :\t',metrics.accuracy_score(y_prediction_flip, actual_bold_flip))
cm = confusion_matrix(actual_bold_flip, y_prediction_flip)
print("Accuracy and Confusion Matrix for Flipped Data")
print(cm)
print("True Negative: " + str(cm[0, 0]))
print("True Positive: " + str(cm[1, 1]))
print("Prediction: Not. Actual: Bold: " + str(cm[1, 0]))
print("Prediction: Bold. Actual: Not: " + str(cm[0, 1]))
count = 0
for i in actual_bold_flip:
    if i == True:
        count +=1
print("actual number of bolded charachters in test: " + str(count))
print("percentage of bold letters detected: " + str(float(cm[1,1]/count)*100))
print("recall: " + str(metrics.recall_score(actual_bold_flip, y_prediction_flip)))
print("precision: " + str(metrics.precision_score(actual_bold_flip, y_prediction_flip)))
print("accuracy score: " + str(accuracy_score(actual_bold_flip, y_prediction_flip)))


# In[9]:


shap.initjs()

explainer_flip = Explainer(model_flip)
shap_values_flip = explainer_flip(X_train_flip)
shap.plots.bar(shap_values_flip, show=True, max_display=16)
shap.plots.beeswarm(shap_values_flip, max_display=16)
for i in range(len(important_features)):
  shap.dependence_plot(i, shap_values_flip.values, X_train_flip, interaction_index=None)


# In[12]:


two_cols[['line width', 'Bolded']].corr()
# two_cols[['rel_top', 'rel_bottom']].corr()


# In[3]:





# In[ ]:





# In[ ]:





# In[7]:


# pic_bold_cors_height = []
#     # base = 10
# b = 2
# for rows in two_cols.itertuples():
#     # print(rows)
#     # print(rows[b])
#     name = rows[b] + "-" + str(rows[b+1]).zfill(3) + ".tif"
#     # print(name)
#     if (rows[b] == "mishivdavar"):
#         name = rows[b] + "-" + str(rows[b+1]).zfill(2) + ".tif"
#     if (rows[b] == "zikukindenura"):
#         name = rows[b] + "-" + str(rows[b+1]) + ".tif"
#     pic = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + name
#     cor = ["r"+str(rows[13]), rows[15], rows[14], rows[16]]
#     pic_bold_cors_height.append([pic, rows[b+3], cor, rows[b+15]])
# print(pic_bold_cors_height[3])


# In[ ]:


# prev_pic = (pic_bold_cors_height[0])[0]
# output = prev_pic.replace("processed_images\processed_images", "height pics")
# print(output)
# print(prev_pic)
# img = Image.open(prev_pic).convert('RGBA')
# draw = ImageDraw.Draw(img)
# print("creted drw")
# for a in pic_bold_cors_height:
#     print(a)
#     if a[0] != prev_pic:
#         output = prev_pic.replace("processed_images\processed_images", "height pics")
#         img.save(output)
#         img.close()
#         print("Next Pic: " + a[0])
#         img = Image.open(a[0]).convert('RGBA')
#         prev_pic = a[0]
#         draw = ImageDraw.Draw(img)
#     # img = Image.open(a[0]).convert('RGBA')
#     if (a[1] == True):
#         if (a[3] <= 10):
#             draw.rectangle(a[2], outline='green', width=1)
#         else:
#             draw.rectangle(a[2], outline='red', width=1)
#     elif (a[1] == False):
#         draw.rectangle(a[2], outline='black', width=1)
# output = prev_pic.replace("processed_images\processed_images", "height pics")
# img.save(output)
# img.close()


# In[ ]:


# index = end
# pic_cors = []
# for i in y_prediction_flip:
#     print(i)
#     if (i == True):
#         name = two_cols['Book Name'][index] + "-" + str(two_cols['Page Number'][index]).zfill(3) + ".tif"
#         pic = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + name
#         cor = [two_cols['Left'][index], two_cols['Top'][index], two_cols['Right'][index], two_cols['Bottom'][index]]
#         print(two_cols.iloc[index])
#     index += 1


# In[ ]:


# cm1 = get_confusion_matrix(model, Pool(X_train, y_train))
# print(cm1)


# In[ ]:


# Now try flipping training and testing data
 # Now test until index 91731 through imreeshresponsa 10
# end = 91731
# X_test_flip = X[:end]
# X_train_flip = X[end+1:]
# y_train_flip = X[end+1:]
# actual_bold_flip = X[:end]
# model_flip = CatBoostClassifier(custom_metric=['AUC:hints=skip_train~false'])
# model_flip.fit(X_train_flip, y_train_flip)
# y_pred_flip = model_flip.predict(X_test_flip)
# y_prediction_flip = [ True if x == "True" else False for x in y_pred_flip]
# print('The accuracy of the catboost flipped model is :\t',metrics.accuracy_score(y_prediction_flip, actual_bold_flip))

