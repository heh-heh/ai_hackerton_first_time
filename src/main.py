import pandas as pd
import random
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train.head()

train_x = train.drop(columns=['id', 'target'])
train_y = train['target']
test_x = test.drop(columns=['id'])

le = LabelEncoder()
le = le.fit(train_x['snowing'])
train_x['snowing'] = le.transform(train_x['snowing'])

for label in np.unique(test_x['snowing']):
    if label not in le.classes_:
        le.classes_ = np.append(le.classes_, label)
    test_x['snowing'] = le.transform(test_x['snowing'])
print('Done.')


LR = LinearRegression()
print('Done.')

poly = PolynomialFeatures(degree=6)
train_x_poly = poly.fit_transform(train_x)

LR.fit(train_x_poly, train_y)


preds = LR.predict(train_x_poly)
print('Done.')

submit = pd.read_csv('./sample_submission.csv')

submit['target'] = preds
submit.head()

submit.to_csv('./submit.csv', index=False)

print('결정계수', LR.score(train_x_poly, train_y))