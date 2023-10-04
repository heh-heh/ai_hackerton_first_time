import pandas as pd
import random
import os
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# seed_everything(42)
train = pd.read_csv('./csv/train.csv')
test = pd.read_csv('./csv/test.csv')

<<<<<<< HEAD:main_/src/main.py
train = pd.read_csv('./open/train.csv')
test = pd.read_csv('./open/test.csv')
=======
>>>>>>> 2aea2f54c0582497dac0afacab210915928389c8:src/main.py

train.head()

train_x = train.drop(columns=['id', 'target'])
train_y = train['target']
test_x = test.drop(columns=['id','target'])

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

<<<<<<< HEAD:main_/src/main.py
=======

first_time = time.time()
>>>>>>> 2aea2f54c0582497dac0afacab210915928389c8:src/main.py
poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)

LR.fit(train_x_poly, train_y)
print('Done.')
lost_time = time.time()

test_x_poly = poly.fit_transform(test_x)
preds = LR.predict(test_x_poly)
print('Done.')

<<<<<<< HEAD:main_/src/main.py
submit = pd.read_csv('./open/sample_submission.csv')
=======
>>>>>>> 2aea2f54c0582497dac0afacab210915928389c8:src/main.py


submit = pd.read_csv('./csv/sample_submission.csv')
submit2 = pd.read_csv('./csv/test2.csv')


# submit['target'] = preds
# submit.head()


submit2['target'] = preds
submit2.head()
submit2.to_csv('./submit.csv', index=False)


fainal = pd.read_csv('./submit.csv')
print('결정계수', LR.score(train_x_poly, train_y))
print('ai 학습 시간', lost_time - first_time)
# print('가중치 : ', LR.coef_)
print('y절편 : ', LR.intercept_)
