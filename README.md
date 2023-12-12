연암공대 풍력발전량 예측 ai해커톤
==============
연암공대에서 주최하고 데이콘이라는 플랫폼에서 진행한 연암공대 재학생, 고등학생 대상으로 진행된 ai해커톤


-----------
![python](https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white)


파일구조
-----------
>data
>>학습데이터와 테스트 데이터 백업

>src
>>main.py : 메인 소스
>>
>>sample_submission.csv : 결과 예측후 저장할 형식
>>
>>test.csv : 결과 예측 할 파일
>>
>>train.csv : 학습용 파일
>>

소스 설명
--------------

머신러닝을 위해 제일 먼제 필요한 모듈부터 불러 오겠습니다.
```py
import pandas as pd ##데이터 로드와 처리를 위한 pandas

from sklearn.model_selection import train_test_split ##데이터 셋을 분리해줌

from sklearn.preprocessing import PolynomialFeatures ##다항 연산을 위해 다항으로 변환 시켜줄 친구
from sklearn.linear_model import LinearRegression ##머신 러닝 모델
```

--------------------

pandas 를 사용해 필요한 파일 불러오기
```py
train = pd.read_csv('./main_heckrton/data/train.csv')
test = pd.read_csv('./main_heckrton/data/test.csv')
```

--------------------

모듈과 데이터를 불러 왔으니 데이터를 가공 해봅시다.
데이터 상관관계를 먼져 보겠습니다.

![이미지 설명](https://github.com/heh-heh/ai_hackerton_first_time/blob/main/Untitled.png)


비교적 발전량과 관계가 없는 강설량, 강수량, 구름_밀집도, 풍향 을 x_data 와 test_x 에서 없애도록 하겠습니다.
```py
x_data = train.drop(['ID', '날씨_주요_요소','날씨_상세_설명', '풍력_발전량','강설량','강수량','구름_밀집도','풍향'], axis=1)
y_data = train['풍력_발전량']

test_x = test.drop(['ID','날씨_주요_요소','날씨_상세_설명','강설량','강수량','구름_밀집도','풍향'],axis=1)
```

---------------------

다항 연산을 위해 다항으로 변환 & 테스트 데이터와 학습데이터 분리
```py
poly = PolynomialFeatures(degree=4)
x_data= poly.fit_transform(x_data)
test_x_poly = poly.transform(test_x)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size = 0.2,shuffle=True, random_state=42)
```

--------------------

모델 정의 + 모델 학습
```py
model = LinearRegression(n_jobs=-1) ##모든 cpu 코어 사용을 위해 n_job 를 -1로 지정

model.fit(X_train, y_train)
```

--------------------

결과 예측 + 저장
```py
res = model.predict(test_x_poly)
sub = pd.read_csv('./main_heckrton/data/sample_submission.csv')
sub['풍력_발전량'] = res
sub.to_csv('./main_heckrton/data/submit.csv', index=False)
```

