import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error
# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)


train = pd.read_csv('./main_heckrton/data/train.csv')
test = pd.read_csv('./main_heckrton/data/test.csv')
train = train.drop(['날씨_주요_요소','날씨_상세_설명'],axis = 1)
test = test.drop(['날씨_주요_요소','날씨_상세_설명'],axis = 1)

train_x = train.drop(columns=['ID', '풍력_발전량'])
train_y = train['ID']
test_x = test.drop(columns=['ID'])




LR = RandomForestRegressor(random_state=156)

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.3, shuffle = True, random_state = 1004 )

LR.fit(X_train, y_train)
print("LR : ", LR.score(X_train,y_train))
print('LR fit Done.')

pra =  {
        'n_estimators':[40,50,80,100],
        'max_depth' : [None], 
        'min_samples_leaf' : [2],
        'min_samples_split' : [1]
        }
gride = GridSearchCV(LR, param_grid=pra,cv=2, refit=True)
gride.fit(X_train,y_train)

print("grid :", gride.best_score_)
print("grid :", gride.score(train_x,train_y))

preds = LR.predict(test_x)
print('Done.')


submit = pd.read_csv('./main_heckrton/data/sample_submission.csv')

submit['풍력_발전량'] = preds

submit.to_csv('./main_heckrton/data/submit.csv', index=False)
