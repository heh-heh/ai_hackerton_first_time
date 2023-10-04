import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso


train = pd.read_csv('./main_heckrton/data/train_test.csv')
test = pd.read_csv('./main_heckrton/data/test.csv')

x_data = train.drop(['ID','날씨_주요_요소','날씨_상세_설명'], axis=1)
y_data = train['e']

test_x = test.drop(['ID','날씨_주요_요소','날씨_상세_설명','강설량','풍향','강수량'],axis=1)




X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size = 0.2, random_state=42)

print(x_data.corr())

##heatmap(data = train.drop(['ID'], axis=1).corr(method='pearson'),
#           annot=True, fmt='.2f',
#           linewidths=.5,
#           cmap='Blues')






# X_train2, X_test, y_train2, y_test = train_test_split(x_data, y_data,test_size = 0.99,shuffle = True, random_state=42)

# plt.scatter(X_train2['기압'], y_train2,alpha=0.4)
# plt.xlabel("target")
# plt.ylabel("Predicted Rent")
# plt.title("MULTIPLE LINEAR REGRESSION")
# plt.show()

# model = Lasso(alpha=0.01)

# poly = PolynomialFeatures(degree=3)
# X_train_poly = poly.fit_transform(X_train)
# test_x_poly = poly.fit_transform(test_x)
# print("poly done")


# model.fit(X_train_poly, y_train)
# print('model done')
# print("model : " , model.score(X_train_poly,y_train))




# res = model.predict(test_x_poly)

# sub = pd.read_csv('./main_heckrton/data/sample_submission.csv')
# sub['풍력_발전량'] = res
# sub.to_csv('./main_heckrton/data/submit2.csv', index=False)



