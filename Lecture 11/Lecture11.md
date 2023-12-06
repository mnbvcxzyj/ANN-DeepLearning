# #11 데이터 다루기

## 딥러닝 & 데이터

<aside>
💡 <b>흐름</b> <br/>
  선형회귀, 로지스틱 회귀 → Perceptron → MLP → 인공신경망(심층신경망)
</aside>

- 좋은 데이터를 만들기 위해 전처리 과정 중요!!

## 피마 인디언 데이터 분석
- 속성 Feature : 8개
- 샘플 수 : 768개
- 클래스 : 1개 (당뇨병 여부 - 정상 0, 당뇨 1)

## Pandas 활용

데이터 전처리를 위해 데이터를 판다스 라이브러리를 통해 시각화 하여 확인

1. 라이브러리, 데이터 불러오기
   - 데이터프레임 형태로 저장

```python
import pandas as pd
import matplotlib.pyplot as plt
import seabron as sns

# 데이터 불러오기
df = pd.read_csv("./pima-indians-diabetes3.csv")
```

1. `value_counts()` : 정상, 당뇨환자 조사 미리 보기
   - df[컬럼명].value_counts() : 각 컬럼 값 개수

```python
df["diabetes"].value_counts()
```

- 정상 500명, 당뇨 268명

1. `describe()` : 정보별 특징
   - 샘플 수, 평균, 표준편차, 최솟값, 백분위 수 해당 값, 최댓값 요약

```python
df.describe()
```


1. `corr()` : 상관관계

```python
df.corr()
```


> 양의 상관관계 분석해야함 

1. 상관관계 그래프 그리기
   - heatmap : 어떤 패턴으로 변화하는지 확인하는 함수
   - vmax : 색상 밝기
   - cmap : 미리 정해진 색상값

```python
# 그래프 색상 구성
colormap = plt.cm.gist_heat
# 그래프 크기
plt.figure(figsize=(12,12))

# 그래프 표시
sns.heatmap(df.corr(), linewidth=0.1, vmax=0.5, cmap=colormap, linecolor="white", annot=True)
plt.show()
```

- 색이 연한게 굿 (1에 가까울 수록 상관도가 높음)

1. 상관도가 높은 plasma로 그려보기
   - 가져오려는 컬럼을 hist()함수의 x축으로 가져옴
     - plasma 칼럼 중 0과 1인 값으로 구분해 불러오게 함
   - bins
     - x축 막대 개수
   - histtype = barstacked
     - 막대바 생성 옵션

```python
plt.figure()
plt.hist(x=[df.plasma[df.diabetes==0], df.plasma[df.diabetes==1]],
				bins=30, histtype="barstacked", label=["normal", "diabetes"])
plt.xlabel("plasma")
plt.ylabel("counts")
plt.grid(True)
plt.legend()
plt.show()
```


1. bmi 기준으로 비율 분포 알아보기

```python
plt.figure()
plt.hist(x=[df.bmi[df.diabetes==0], df.bmi[diabetes==1]],
		bins = 30, histtype="barstacked", label=["normal", "diabetes"])
plt.xlabel("bmi")
plt.ylabel("counts")
plt.grid(True)
plt.legend()
plt.show()
```

## 피마 인디언 당뇨병 예측 인공신경망 모델 설계

- `iloc[:,]`
  - 대괄호 안에 정한 범위만큼 가져와 저장

```python
df = pd.read_csv("./data/pima-indians-diabetes3.csv")

# 세부 정보
x = df.iloc[:, 0:8]

# 당뇨병 여부
y = df.iloc[:, 8]
```

- 모델 구조

```python
model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(12, activation="relu", name="Dense1"))
model.add(Dense(8, activation="relu", name="Dense2"))
model.add(Dense(1, activation="sigmoid", name="Dense3"))
model.summary()
```


[model.summary]

1. Layer
   - 각 층의 이름과 유형
2. Output Shape
   - 각 층에 몇 개의 출력이 발생하는지 나타냄
   - 샘플수, 속성 수
   - 8개 입력 → 12개 → 8개 → 1개 출력
3. Param
   - 파라미터 수 (가중치 + 바이어스 합)
   - 8개 \* 12 = 96 + 바이어스 12 = 108
4. 요약
   - 전체 파라미터 합산

