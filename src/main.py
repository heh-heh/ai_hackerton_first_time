

##필요한 모듈 불러오기
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error



train = pd.read_csv('./main_heckrton/data/train.csv')
test = pd.read_csv('./main_heckrton/data/test.csv')

x_data = train.drop(['ID', '날씨_주요_요소','날씨_상세_설명', '풍력_발전량','강설량','강수량','구름_밀집도','풍향'], axis=1)
y_data = train['풍력_발전량']

test_x = test.drop(['ID','날씨_주요_요소','날씨_상세_설명','강설량','강수량','구름_밀집도','풍향'],axis=1)

poly = PolynomialFeatures(degree=4)
x_data= poly.fit_transform(x_data)
test_x_poly = poly.transform(test_x)
print("poly done")

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size = 0.2,shuffle=True, random_state=42)

model = LinearRegression(n_jobs=-1) 

model.fit(X_train, y_train)
print('model done')
print("model (transform): " , model.score(X_train,y_train))

res = model.predict(test_x_poly)
sub = pd.read_csv('./main_heckrton/data/sample_submission.csv')
sub['풍력_발전량'] = res
sub.to_csv('./main_heckrton/data/submit.csv', index=False)





