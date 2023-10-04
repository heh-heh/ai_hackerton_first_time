import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso


train = pd.read_csv('./main_heckrton/data/train.csv')
test = pd.read_csv('./main_heckrton/data/test.csv')

x_data = train.drop(['ID', '날씨_상세_설명','날씨_주요_요소', '풍력_발전량','강설량','풍향','강수량','구름_밀집도'], axis=1)
y_data = train['풍력_발전량']

test_x = test.drop(['ID','날씨_주요_요소','날씨_상세_설명','강설량','풍향','강수량','구름_밀집도'],axis=1)




X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size = 0.2, random_state=42)
# X_train2, X_test, y_train2, y_test = train_test_split(x_data, y_data,test_size = 0.99,shuffle = True, random_state=42)

# plt.scatter(X_train2['기압'], y_train2,alpha=0.4)
# plt.xlabel("target")
# plt.ylabel("Predicted Rent")
# plt.title("MULTIPLE LINEAR REGRESSION")
# plt.show()

model = LinearRegression(n_jobs=-1) 
model2 = Lasso(alpha=0.01)

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
test_x_poly = poly.fit_transform(test_x)
print("poly done")


model.fit(X_train_poly, y_train)
print('model done')
print("model : " , model.score(X_train_poly,y_train))




res = model.predict(test_x_poly)

sub = pd.read_csv('./main_heckrton/data/sample_submission.csv')
sub['풍력_발전량'] = res
sub.to_csv('./main_heckrton/data/submit.csv', index=False)



