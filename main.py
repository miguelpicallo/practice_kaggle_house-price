# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 11:22:09 2021

@author: mpica
"""
# 0. import packages
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV,Lasso, LinearRegression
from sklearn.feature_selection import f_regression, mutual_info_regression
from xgboost import XGBRegressor
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
%matplotlib inline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%


# 1. read data
df_train = pd.read_csv('C:/Users/mpica/Pictures/Documentos/kaggle/house-price2/train.csv',index_col = 'Id')
df_test = pd.read_csv('C:/Users/mpica/Pictures/Documentos/kaggle/house-price2/test.csv',index_col = 'Id')

# 2. explore data
print(df_train.dtypes)
print(df_train.head())

#%%
# 3. impute NAs
df_train['MSSubClass'].value_counts()
df_train['MSSubClass'] = df_train['MSSubClass'].astype('object')
df_test['MSSubClass'] = df_test['MSSubClass'].astype('object')

print(df_train.isna().sum())
print(df_test.isna().sum())

col_objects = df_train.columns[df_train.dtypes == 'object']
col_num = df_test.columns[df_test.dtypes != 'object']
for col in col_objects:
    df_train.loc[df_train[col].isna(),col] = 'None'
    df_test.loc[df_test[col].isna(),col] = 'None'
for col in col_num:
    df_train.loc[df_train[col].isna(),col] = 0
    df_test.loc[df_test[col].isna(),col] = 0
    
# df_train = df_train.dropna()

#%%
# 4. Address ordinal variables
df_train['new_Residential'] = df_train['MSZoning'].map(lambda x: x in ['RH','RL','RP','RM']).astype('int64')
df_train['new_CentralAir'] = df_train['CentralAir'].map(lambda x: x in ['Y']).astype('int64')

df_test['new_Residential'] = df_test['MSZoning'].map(lambda x: x in ['RH','RL','RP','RM']).astype('int64')
df_test['new_CentralAir'] = df_test['CentralAir'].map(lambda x: x in ['Y']).astype('int64')

ordEncode_col = ['LotShape']
ordEncode_categories = [['None','Reg','IR1','IR2','IR3']]
ordEncode_col.append('LandContour')
ordEncode_categories.append(['None','Low','Lvl','Bnk','HLS'])
ordEncode_col.append('Utilities')
ordEncode_categories.append(['None','ELO','NoSeWa','NoSewr','AllPub'])
ordEncode_col.append('LandSlope')
ordEncode_categories.append(['None','Gtl','Mod','Sev'])
ordEncode_col.append('ExterQual')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('ExterCond')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('BsmtQual')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('BsmtCond')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('BsmtExposure')
ordEncode_categories.append(['None','No','Mn','Av','Gd'])
ordEncode_col.append('BsmtFinType1')
ordEncode_categories.append(['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'])
ordEncode_col.append('BsmtFinType2')
ordEncode_categories.append(['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'])
ordEncode_col.append('HeatingQC')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('KitchenQual')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('Functional')
ordEncode_categories.append(['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal','None'])
ordEncode_col.append('FireplaceQu')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('GarageFinish')
ordEncode_categories.append(['None','Unf','RFn','Fin'])
ordEncode_col.append('GarageQual')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('GarageCond')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('PavedDrive')
ordEncode_categories.append(['None','N','P','Y'])
ordEncode_col.append('PoolQC')
ordEncode_categories.append(['None','Po','Fa','TA','Gd','Ex'])
ordEncode_col.append('Fence')
ordEncode_categories.append(['GdPrv', 'MnPrv', 'GdWo', 'MnWw','None'])

ordEncode = OrdinalEncoder(categories = ordEncode_categories)
col_num = df_test.columns[df_test.dtypes != 'object']
ordEncoded_train = pd.DataFrame(ordEncode.fit_transform(df_train[ordEncode_col]))
ordEncoded_test = pd.DataFrame(ordEncode.transform(df_test[ordEncode_col]))
ordEncoded_train.columns = ordEncode_col
ordEncoded_test.columns = ordEncode_col
ordEncoded_train.index = df_train.index
ordEncoded_test.index = df_test.index

X_train = pd.concat([df_train[col_num], ordEncoded_train],axis=1)
X_test = pd.concat([df_test[col_num], ordEncoded_test],axis=1)
y_train = np.log(df_train['SalePrice'])

sns.displot(df_train['SalePrice'])
sns.displot(np.log(df_train['SalePrice']))

#%%
# 5. model and validation

# random forest
scores_all = []
for n_estimators in [10,50,100,200]:
    scores = -1*cross_val_score(RandomForestRegressor(n_estimators = n_estimators, min_samples_leaf=10, random_state=0),
                            X_train,y_train,cv=10,scoring='neg_mean_squared_error')
    scores_all.append(np.mean(scores))
print(scores_all)
    
scores_all = []
for min_leaf in [1,2,5,10]:
    scores = -1*cross_val_score(RandomForestRegressor(n_estimators = 100, min_samples_leaf=min_leaf, random_state=0),
                            X_train,y_train,cv=10,scoring='neg_mean_squared_error')
    scores_all.append(np.mean(scores))
print(scores_all) 

model = RandomForestRegressor(n_estimators = 100, min_samples_leaf=2, random_state=0)
model.fit(X_train,y_train)
print(X_train.columns[np.argsort(model.feature_importances_).tolist()][::-1])

# Lasso
scores_all = []
for alpha in np.logspace(-5, 1, 7):
    scores = -1*cross_val_score(Lasso(random_state=0,alpha=alpha),X_train,y_train,cv=10,scoring='neg_mean_squared_error')
    scores_all.append(np.mean(scores))
print(scores_all) 

scores = -1*cross_val_score(LassoCV(),X_train,y_train,cv=10,scoring='neg_mean_squared_error')
print(np.mean(scores)) 
model = LassoCV(random_state=0, cv=10)
model.fit(X_train,y_train)

#light GBM
scores_all = []
for n_estimators in [50,100,200, 300, 500]:
    scores = -1*cross_val_score(HistGradientBoostingRegressor(loss='squared_error', 
                            learning_rate=0.1, max_iter= n_estimators, min_samples_leaf=10, 
                            random_state=0),
                            X_train,y_train,cv=10,scoring='neg_mean_squared_error')
    scores_all.append(np.mean(scores))
    print(n_estimators, np.mean(scores), np.std(scores))
print(scores_all) 

# test several models
models = {'randomforest': RandomForestRegressor(n_estimators = 100, min_samples_leaf=2, random_state=0),
          'boosting tree': HistGradientBoostingRegressor(max_iter = 200, random_state=0),
          'lasso linear': LassoCV(random_state=0, cv=10, alphas = np.logspace(-5, 1, 7).tolist())}
scores_all = []
for model_name in models:
    scores = -1*cross_val_score(models[model_name],X_train,y_train,cv=10,scoring='neg_mean_squared_error')
    scores_all.append(np.mean(scores))
print(scores_all)

#%% 
# 6. statistical inference
X_train_pIndep = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_pIndep).fit()
model.summary()

# model.fittedvalues = model.predict(X_train_pIndep)
# model.resid = y_train-model.fittedvalues
# model.rsquared = 1-sum(model.resid**2)/sum((y_train-np.mean(y_train))**2)
# model.rsquared_adj = 1-sum(model.resid**2)/sum((y_train-np.mean(y_train))**2)*(model.nobs-1)/model.df_resid
# model.mse_total = 
# model.mse_resid = sum(model.resid**2)/model.df_resid
# model.mse_model = (sum((y_train-np.mean(y_train))**2)-sum(model.resid**2))/model.df_model
# model.mse_model = sum((model.fittedvalues-np.mean(y_train))**2)/model.df_model
# F-stats = model.mse_model/model.mse_resid


model2 = sm.GLM(y_train, X_train_pIndep, family=sm.families.Gaussian()).fit()
model2.summary2()
# model2.deviance = sum(model2.resid_deviance**2)

# recover F-statistic
model2.f_test(np.identity(len(model2.params))[1:,:])
#deviance = sum(model2.resid_response**2)
#pearsonchi2 = sum(model2.resid_pearson**2)

# test continuous to continuous variables: univariate analysis
f_val,p_val = f_regression(X_train,y_train)

#%% 
# 7. test stuff

# 7.1 colinearity
n = 100
x1 = np.random.normal(0,1,n)
x2 = 2*x1 +2 + np.random.normal(0,0.1,n)
y = 3*x1 + 1 + np.random.normal(0,0.1,n)
c = np.ones(n)

data = pd.DataFrame({'x1':x1,'x2':x2, 'c':c, 'y':y})

model = smf.glm(formula='y~c+x1-1',data=data, family=sm.families.Gaussian()).fit()
model2 = smf.glm(formula='y~x1',data=data, family=sm.families.Gaussian()).fit()
model3 = smf.glm(formula='y~x1+x2',data=data, family=sm.families.Gaussian()).fit()
print(model.summary())

M=np.array(data[['c','x1','x2']])
print(np.linalg.eig(M.T.dot(M)))

# 7.2 coin to estimate pi
n = 10000
h = 10000
# 0: just sample uniforms, then avg with norm <1 =pi/4
UX = stats.randint.rvs(0,h+1,size=n)/h *2-1
UY = stats.randint.rvs(0,h+1,size=n)/h *2-1
print(sum((UX**2+UY**2)<1)/n*4)
# 1: generate uniform [-1,1]
X = stats.binom.rvs(h,0.5,size=n)
Y = stats.binom.rvs(h,0.5,size=n)
cdfY = stats.binom.cdf(Y,h,0.5)
cdfX = stats.binom.cdf(X,h,0.5)
UY = cdfY*2-1
UX = cdfX*2-1
print(sum((UX**2+UY**2)<1)/n*4)
# 2: importance sampling, sample binomial n, correct with randint [0,n]
pmfX = stats.binom.pmf(X,h,0.5)
pmfY = stats.binom.pmf(Y,h,0.5)
# BX =X/h*2-1
# BY =Y/h*2-1
# print(sum(1.*((BX**2+BY**2)<1)/pmfX/pmfY/(h+1)/(h+1))/n*4)
print(sum(1.*( ((X/h*2-1)**2+(Y/h*2-1)**2)<1 )/pmfX/pmfY/(h+1)/(h+1))/n*4)
# approx with normal
pmfnormX = stats.norm.pdf(BX,0,np.std(BX))
pmfnormY = stats.norm.pdf(BY,0,np.std(BY))
print(sum(1.*((BX**2+BY**2)<1)/pmfnormX/pmfnormY/2/2)/n*4)

# test example
X = stats.norm.rvs(0.5,1,size=n)
sum(X*stats.uniform.pdf(X,0,1)/stats.norm.pdf(X,0.5,1))/n # 0.5
sum(X**2*stats.uniform.pdf(X,0,1)/stats.norm.pdf(X,0.5,1))/n # 0.5**2 + 1./12
X = stats.binom.rvs(9,0.5,size=n)
sum((X<5)*stats.randint.pmf(X,0,10)/stats.binom.pmf(X,9,0.5))/n # 0.5

# 7.3 stratified sampling
X1 = stats.norm.rvs(0,1,size = 100)
X2 = stats.norm.rvs(10,4,size = 200)
Y = np.concatenate([X1,X2])
print(np.mean(Y))

mean1 = []; mean2=[]; mean3=[]
for i in range(50):
    # same # samples
    X1s = X1[np.random.choice(len(X1),15)]
    X2s = X2[np.random.choice(len(X2),15)]
    mean1.append(np.mean(X1s)/3+np.mean(X2s)*2./3)
    # proportional population
    X1s = X1[np.random.choice(len(X1),10)]
    X2s = X2[np.random.choice(len(X2),20)]
    mean2.append(np.mean(X1s)/3+np.mean(X2s)*2./3)
    # optimal variance
    X1s = X1[np.random.choice(len(X1),6)]
    X2s = X2[np.random.choice(len(X2),24)]
    mean3.append(np.mean(X1s)/3+np.mean(X2s)*2./3)
print(np.mean(mean1),np.mean(mean2),np.mean(mean3),np.std(mean1),np.std(mean2),np.std(mean3))
