# ai_hackerton_first_time
연암공대 풍력발전량 예측 해커톤 참가하여 작성한 소스

파일구조
-----------
>data
>>학습데이터와 테스트 데이터 백업

>src
>>main.py : 메인 소스
>>sample_submission.csv : 결과 예측후 저장할 형식
>>test.csv : 결과 예측 할 파일
>>trin.csv : 학습용 파일

소스 설명
--------------
필요한 모듈 불러오기
```py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
```

pandas 를 사용해 필요한 파일 불러오기
```py
train = pd.read_csv('./main_heckrton/data/train.csv')
test = pd.read_csv('./main_heckrton/data/test.csv')
```

테스트 데이터와 학습 데이터 불필요한 요소 제거
```py
x_data = train.drop(['ID', '날씨_주요_요소','날씨_상세_설명', '풍력_발전량','강설량','강수량','구름_밀집도','풍향'], axis=1)
y_data = train['풍력_발전량']

test_x = test.drop(['ID','날씨_주요_요소','날씨_상세_설명','강설량','강수량','구름_밀집도','풍향'],axis=1)
```

다항 연산을 위해 다항으로 변환 & 테스트 데이터와 학습데이터 분리
```py
poly = PolynomialFeatures(degree=4)
x_data= poly.fit_transform(x_data)
test_x_poly = poly.transform(test_x)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size = 0.2,shuffle=True, random_state=42)
```

모델 정의 + 모델 학습
```py
model = LinearRegression(n_jobs=-1) 

model.fit(X_train, y_train)
```

결과 예측 + 저장
```py
res = model.predict(test_x_poly)
sub = pd.read_csv('./main_heckrton/data/sample_submission.csv')
sub['풍력_발전량'] = res
sub.to_csv('./main_heckrton/data/submit.csv', index=False)
```

