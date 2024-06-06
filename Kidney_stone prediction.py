#!/usr/bin/env python
# coding: utf-8

# ## 0. Loading Data and Libraries

# In[58]:


# IMPORTING AND LOADING LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('ggplot')
import sklearn
import xgboost as xgb
import scipy.stats as stats

from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_log_error, make_scorer, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.svm import SVR
from sklearn import svm

import warnings
warnings.filterwarnings('ignore')


# In[59]:


# LOADING DATASET


df = pd.read_csv('/Users/abhishek/Downloads/train.csv')
test_df = pd.read_csv('/Users/abhishek/Downloads/test.csv')
df_og = pd.read_csv('/Users/abhishek/Downloads/kindey stone urine analysis.csv')
dff = test_df
test_df = test_df.drop(['id'], axis=1)


# In[60]:


# VIEWING THE COLUMNS IN THE DATASET

df.dtypes


# In[61]:


df.shape


# In[62]:


df.describe().T


# In[63]:


#Checking missing values

df.isna().sum()


# In[64]:


# LOOKING FOR DUPLICATES

df.loc[df.duplicated()]


# In[65]:


df.head()


# ## 1. EDA and Visualisation

# In[66]:


columns_bp = list(df.columns)
columns_bp.remove('id')
columns_bp.remove('target')


# In[67]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
colors = ['#4CAF50','#9E9E9E', '#9E9E9E', '#4CAF50', '#4CAF50','#9E9E9E','#9E9E9E', '#4CAF50']

for i, ax in enumerate(axs.flatten()):
    col_name = columns_bp[i]
    color = colors[i]
    bp = ax.boxplot(df[col_name], patch_artist=True, medianprops=dict(color='black'))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
    ax.set_title(f'{col_name} BoxPlot: ')

plt.show()


# In[68]:


columns = list(df.columns)

columns.remove('id')

cat = 0
num = 0

for i in range(7):
    if df[columns[i]].nunique() <= 10:
        cat+=1
        
    else:
        num+=1
        
print('No. of Numerical Variables:', num)
print('No. of Categorical Variables:', cat)


# In[69]:


num_df = pd.DataFrame()

for col in df.columns:
    if df[col].nunique() > 10:
        num_df[col] = df[col]
        
num_df = num_df.drop('id', axis=1)


# In[70]:


columns_num = list(num_df.columns)


for i in range(5):
    fig_num, ax2 = plt.subplots()
    ax2.hist(num_df[columns_num[i]], bins=50, density=True, color='coral')
    ax2.set_title(f'{columns_num[i]} distribution')
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    kde = stats.gaussian_kde(num_df[columns_num[i]])
    ax2.plot(x, kde(x), color='navy')
    plt.show()


# In[71]:


df['target'].value_counts().plot(kind='bar', color = '#0075A2')

plt.xlabel('Categoriies')
plt.ylabel('Count')
plt.title('Target Bar Plot')

plt.show()


# In[72]:


import dtale


# In[23]:


dtale.show(df)


# In[73]:


# calc is skewed so needs transformation?

df['log_calc'] = np.log(df['calc'])


# In[74]:


plt.hist(df['log_calc'])


# In[75]:


df = df.drop(['log_calc'], axis = 1)


# In[76]:


dtale.show(df)


# In[77]:


corr_matrix = df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 2. Feature Selection and Engineering

# In[78]:


df = df.drop('id', axis=1)


# In[79]:


df_merged = pd.concat([df, df_og], ignore_index=True, sort=False)


# In[ ]:





# In[93]:


df_merged['cond/osmo'] = df_merged['cond']/df_merged['osmo']
df_merged['calc/ph'] = df_merged['calc']/df_merged['ph']


# In[ ]:





# In[104]:


test_df['cond/osmo'] = test_df['cond']/test_df['osmo']
test_df['calc/ph'] = test_df['calc']/test_df['ph']


# In[ ]:





# In[ ]:





# ## 3. Model building and Assessment

# In[106]:


X_train, X_test, y_train, y_test = train_test_split(df_merged.drop(['target'], axis=1), df_merged['target'], test_size=0.2, random_state=42)


# In[107]:


X_train.isna().sum()


# In[108]:


rf = RandomForestClassifier(n_estimators=100,
                            max_depth = 3,
                            random_state=42)


# In[109]:


xgb_cl = xgb.XGBClassifier(n_estimators = 100, 
                          learning_rate = 0.05, 
                          max_depth = 1,
                          eval_metric = "auc")


# In[136]:


voting_classifier = VotingClassifier(estimators=[('rf', rf), ('xgb_cl', xgb_cl)], voting = 'soft',weights=[1, 4] )


# In[137]:


voting_classifier.fit(X_train, y_train)


# In[138]:


voting_auc = roc_auc_score(y_test, voting_classifier.predict_proba(X_test)[:, 1])
print(f"Voting Classifier ROC AUC score: {voting_auc:.2f}")


# In[115]:


voting_classifier.fit(df_merged.drop(['target'], axis=1), df_merged['target'])


# In[ ]:





# In[116]:


predictions = voting_classifier.predict_proba(test_df)[:, 1]


# In[117]:


submission = pd.DataFrame({'id': dff['id'], 'target': predictions})
submission.to_csv('kidneystonepred8.csv', index=False)


# In[ ]:





# In[ ]:





# In[230]:


# assuming y_test and y_pred_proba are already defined
fpr, tpr, thresholds = roc_curve(y_test, voting_classifier.predict_proba(X_test)[:, 1])
roc_auc = roc_auc_score(y_test, voting_classifier.predict_proba(X_test)[:, 1])


# In[231]:


# plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.show()

# print AUC score
print('AUC Score: {:.2f}'.format(roc_auc))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[219]:


xgb_cl.fit(X_train, y_train)


# In[220]:


xgb_auc = roc_auc_score(y_test, xgb_cl.predict_proba(X_test)[:, 1])


# In[221]:


print(f"XGBoost ROC AUC score: {xgb_auc:.2f}")


# In[201]:


rf.fit(X_train, y_train)


# In[202]:


rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"Random Forest ROC AUC score: {rf_auc:.2f}")


# In[232]:


voting_classifier.fit(df_merged.drop(['target'], axis=1), df_merged['target'])


# In[233]:


predictions = voting_classifier.predict_proba(test_df)[:, 1]


# In[235]:


# Saving submission

submission = pd.DataFrame({'id': dff['id'], 'target': predictions})
submission.to_csv('kidneystonepred3.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[245]:


# with cond/osmo and calc/ph

df_merged_1 = df_merged.drop(['cond', 'osmo', 'calc/ph'], axis = 1)


# In[246]:


df_merged_2 = df_merged.drop(['calc', 'ph', 'cond/osmo'], axis = 1)


# In[247]:


df_merged_3 = df_merged.drop(['cond', 'osmo', 'calc', 'ph'], axis = 1)


# In[ ]:





# In[295]:


## testing cond/osmo ##
X_train, X_test, y_train, y_test = train_test_split(df_merged_1.drop(['target'], axis=1), df_merged_1['target'], test_size=0.2, random_state=42)


# In[296]:


voting_classifier.fit(X_train, y_train)


# In[297]:


voting_auc = roc_auc_score(y_test, voting_classifier.predict_proba(X_test)[:, 1])
print(f"Voting Classifier ROC AUC score: {voting_auc:.2f}")


# In[298]:


# assuming y_test and y_pred_proba are already defined
fpr, tpr, thresholds = roc_curve(y_test, voting_classifier.predict_proba(X_test)[:, 1])
roc_auc = roc_auc_score(y_test, voting_classifier.predict_proba(X_test)[:, 1])


# In[299]:


# plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.show()

# print AUC score
print('AUC Score: {:.2f}'.format(roc_auc))


# In[302]:


voting_classifier.fit(df_merged_1.drop(['target'], axis=1), df_merged_1['target'])


# In[303]:


test_df_1 = test_df
test_df_1['cond/osmo'] = test_df['cond']/test_df['osmo']
test_df_1 = test_df_1.drop(['cond', 'osmo'], axis=1)


# In[278]:


predictions_1 = voting_classifier.predict_proba(test_df_1)[:, 1]


# In[281]:


# Saving submission

submission = pd.DataFrame({'id': dff['id'], 'target': predictions_1})
submission.to_csv('kidneystonepred4.csv', index=False)


# In[ ]:





# In[ ]:





# In[257]:


## testing calc/ph ##
X_train, X_test, y_train, y_test = train_test_split(df_merged_2.drop(['target'], axis=1), df_merged['target'], test_size=0.2, random_state=42)


# In[258]:


voting_classifier.fit(X_train, y_train)


# In[259]:


voting_auc = roc_auc_score(y_test, voting_classifier.predict_proba(X_test)[:, 1])
print(f"Voting Classifier ROC AUC score: {voting_auc:.2f}")


# In[260]:


# assuming y_test and y_pred_proba are already defined
fpr, tpr, thresholds = roc_curve(y_test, voting_classifier.predict_proba(X_test)[:, 1])
roc_auc = roc_auc_score(y_test, voting_classifier.predict_proba(X_test)[:, 1])


# In[261]:


# plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.show()

# print AUC score
print('AUC Score: {:.2f}'.format(roc_auc))


# In[263]:


## testing calc/ph and cond/osmo ##

X_train, X_test, y_train, y_test = train_test_split(df_merged_3.drop(['target'], axis=1), df_merged['target'], test_size=0.2, random_state=42)


# In[264]:


voting_classifier.fit(X_train, y_train)


# In[265]:


voting_auc = roc_auc_score(y_test, voting_classifier.predict_proba(X_test)[:, 1])
print(f"Voting Classifier ROC AUC score: {voting_auc:.2f}")


# In[266]:


# assuming y_test and y_pred_proba are already defined
fpr, tpr, thresholds = roc_curve(y_test, voting_classifier.predict_proba(X_test)[:, 1])
roc_auc = roc_auc_score(y_test, voting_classifier.predict_proba(X_test)[:, 1])


# In[267]:


# plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.show()

# print AUC score
print('AUC Score: {:.2f}'.format(roc_auc))


# In[ ]:





# In[ ]:


# constructing new features and testing


# In[304]:


df_merged['calc*ph'] = df_merged['calc']*df_merged['ph']
df_merged['urea*cond'] = df_merged['urea']*df_merged['cond']
df_merged['gravity*calc'] = df_merged['gravity']*df_merged['calc']
df_merged['calc+urea*gravity'] = (df_merged['calc']+df_merged['urea'])*df_merged['gravity']
df_merged['calc+urea+osmo/cond'] = (df_merged['calc']+df_merged['urea']+df_merged['osmo'])/df_merged['cond']
df_merged['gravity+osmo+cond+ph'] = df_merged['gravity']+df_merged['osmo']+df_merged['cond']+df_merged['ph']



# In[315]:


df_merged['gravity/calc'] = df_merged['gravity']/df_merged['calc']


# In[342]:


df_merged


# In[317]:


corr_matrix = df_merged.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()


# In[365]:


X_train, X_test, y_train, y_test = train_test_split(df_merged.drop(['target'], axis=1), df_merged['target'], test_size=0.2, random_state=42) 


# In[435]:


voting_classifier.fit(X_train, y_train)


# In[436]:


voting_auc = roc_auc_score(y_test, voting_classifier.predict_proba(X_test)[:, 1])
print(f"Voting Classifier ROC AUC score: {voting_auc:.2f}")


# In[427]:


rf = RandomForestClassifier(n_estimators=100,
                            max_depth = 3,
                            random_state=42)


# In[428]:


rf.fit(X_train, y_train)


# In[429]:


preds = rf.predict_proba(X_test)
rf_score = roc_auc_score(y_test, preds[:,1])
print("ROC-AUC Score:", rf_score)


# In[402]:


xgb_cl.fit(X_train, y_train)


# In[403]:


preds = xgb_cl.predict_proba(X_test)
xgb_score = roc_auc_score(y_test, preds[:,1])
print("ROC-AUC Score:", xgb_score)


# In[352]:


test_df['cond/osmo'] = test_df['cond']/test_df['osmo']
test_df['calc/ph'] = test_df['calc']/test_df['ph']
test_df['calc*ph'] = test_df['calc']*test_df['ph']
test_df['urea*cond'] = test_df['urea']*test_df['cond']
test_df['gravity*calc'] = test_df['gravity']*test_df['calc']
test_df['calc+urea*gravity'] = (test_df['calc']+test_df['urea'])*test_df['gravity']
test_df['calc+urea+osmo/cond'] = (test_df['calc']+test_df['urea']+test_df['osmo'])/test_df['cond']
test_df['gravity+osmo+cond+ph'] = test_df['gravity']+test_df['osmo']+test_df['cond']+test_df['ph']
test_df['gravity/calc'] = test_df['gravity']/test_df['calc']


# In[356]:


test_df = test_df.drop(['id'], axis = 1)


# In[353]:


test_df


# In[404]:


xgb_cl.fit(df_merged.drop(['target'], axis=1), df_merged['target'])


# In[405]:


predictions = xgb_cl.predict_proba(test_df)[:, 1]


# In[406]:


submission = pd.DataFrame({'id': dff['id'], 'target': predictions})
submission.to_csv('kidneystonepred6.csv', index=False)


# In[ ]:





# In[437]:


voting_classifier.fit(df_merged.drop(['target'], axis=1), df_merged['target'])


# In[438]:


predictions_1 = voting_classifier.predict_proba(test_df)[:, 1]


# In[439]:


submission = pd.DataFrame({'id': dff['id'], 'target': predictions_1})
submission.to_csv('kidneystonepred7.csv', index=False)


# In[ ]:





# In[447]:


svm_cl = svm.SVC(kernel='linear', C=1, probability= True)


# In[ ]:


svm_cl.fit(df_merged.drop(['target'], axis=1), df_merged['target'])


# In[443]:


predictions_2 = svm_cl.predict_proba(test_df)[:, 1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




