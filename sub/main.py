import pandas as pd 
import numpy as np

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pit

from sklearn.model_selection import train_test_split

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')

train = train.drop(columns=['ID'])
test = test.drop(columns=['ID'])

model = RandomForestRegressor()
# train.info()

train = train.drop(columns=['ID'])
test = test.drop(columns=['ID'])


x_train = train.drop(columns=['TARGET'])
y_train = train['TARGET']


model.fit(x_train, y_train)
res = model.predict(test)

sample_submission['TARGET'] = res
sample_submission.to_csv("submission.csv",index=False)

print(model.score(x_train,y_train))


poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
test_poly = poly.fit_transform(test)