# Kaggle: ICR
**Kaggle Competition | ICR** - Identifying Age-Related Conditions

### 사용기술
![Python](https://img.shields.io/badge/Python-ffe74a.svg?style=flat&logo=Python&logoColor=blue) 
![NumPy](https://img.shields.io/badge/NumPy-4d77cf.svg?style=flat&logo=NumPy&logoColor=4dabcf) 
![Pandas](https://img.shields.io/badge/Pandas-130654.svg?style=flat&logo=Pandas&logoColor=whitle) 
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C.svg?style=flat&logo=Matplotlib&logoColor=white) 
![Plotly](https://img.shields.io/badge/Plotly-262626.svg?style=flat&logo=Plotly&logoColor=white) 
![ScikitLearn](https://img.shields.io/badge/Scikit%20Learn-F79939.svg?style=flat&logo=Scikit%20Learn&logoColor=3499CD) 

## OverView

### 대회소개
- 주어진 조건으로 환자가 3가지 노화관련 질병에 해당 되는지 예측합니다.
- 3가지 질병 중 하나 이상을 가지고 있는지(Class 1) 또는 아무 질병도 가지고 있지 않는지(Class 0)로 구분 하는 이진분류 $^{\textrm{Binary Classification}}$ 문제입니다.

### 배경
- 나이는 숫자에 불과하다는 말이 있지만, 노화와 함께 수많은 건강 문제가 발생합니다. 심장병과 치매에서 청력 상실과 관절염에 이르기까지, 노화는 수많은 질병과 합병증의 위험 요소입니다. 생물정보학 분야는 생물학적 노화를 늦추거나 되돌리고 주요 노화 관련 질병을 예방하는 데 도움이 될 수 있는 개입에 대한 연구를 포함하여 성장하고 있습니다. 데이터 과학은 샘플 수가 적더라도 다양한 데이터로 문제를 해결할 수 있는 새로운 방법을 개발하는 데 중요한 역할을 할 수 있습니다.

- 현재 건강상태를 예측하는데 XGBoost나 랜덤포레스트와 같은 모델이 사용되고 있지만, 그 성능이 충분하지 않습니다. 생명이 달린 중요한 문제를 다루기 위해서는 모델이 다양한 사례에 대해 안정적이고 일관되게 정확한 예측을 내려야 합니다.

**참여기간**: 23.06.19 ~ 23.07.05

**참여방식**: 개인

<br>

## 데이터

**input:** 56개 건강 특성 (anonymized health characteristics)

- **Train.csv**: 학습용 데이터. Id, 의료정보가 담긴 features, class의 열로 구성되어 있습니다.
    - `Id` 각 관측값에 대하여 식별하는 고유 ID입니다.
    - `AB-GL` 익명화된 56개의 건강 특성. categorical인 `EJ`를 제외하고 모두 숫자입니다.
    - `class` binary target : `1`은 피험자가 세 가지 조건 중 하나로 진단받았음을 나타내고, `0`은 진단받지 않았음을 나타냅니다.
    - EJ 컬럼은 categorical, 나머지는 모두 numeric 데이터
    - BQ와 EL의 결측치가 60개, 그 외 7개의 컬럼에는 각 1~3개의 결측치가 있음.

- **Test.csv**: 피험자가 두 `class` 각각에 속할 확률을 예측하는 것 입니다. Id와 의료정보가 담긴 features로 구성되어 있습니다.

- **greeks.csv**: 학습용 데이터에만 사용 가능한 보조 metadata.
    - `Alpha` : 노화 여부 코드로 조건이 있는 경우 해당 유형을 식별합니다.
        - `A` : 노화 관련 조건이 없습니다. class `0`에 해당합니다.
        - `B, D, G` : 3가지 실험적 특징 experimental characteristics. class `1`에 해당합니다.
    - `Beta`, `Gamma`, `Delta` : 세 가지 실험 특성입니다.    
    - `Epsilon` : 이 피험자에 대한 데이터가 수집된 날짜입니다. 테스트 세트의 모든 데이터는 훈련 세트가 수집된 후에 수집되었습니다. 결측치는 없지만, Epsilon컬럼 중 ‘Unknown’이 144개 존재합니다.

**output:** 진단 여부 (3가지 중 하나 이상 = class 1, or not = class 0)
- **sample_of_submission.csv** : 제출용 형식의 파일

<br>

## 평가지표
**Balanced logarithmic loss**

$$ Log\ Loss = \frac{-\frac{1}{N_0}\Sigma_{i=1}^{N_0}y_{0i}\log{p_{0i}}-\frac{1}{N_1}\Sigma_{i=1}^{N_1}y_{1i}\log{p_{1i}}}{2} $$

- 환자와 비 환자의 Logloss을 각각 계산하여 그 값의 평균을 반환.
- Logloss는 다른 평가지표와 비교했을 때, 우연히 정답을 맞추는 것을 방지하기 위하여 틀린 값에 대해 페널티를 줌.
- 해당 대회의 학습train 데이터에서 타겟(Class 0, 1) 데이터가 균등하게 분포되지 않아 데이터 더 많은 Class로 예측하게 되는 문제가 있으며, 이를 보정할 수 있는 Balanced logloss로 평가.

<br>

## 탐색적 데이터분석 EDA

<p align='center'><img src="assets/fig00.png" width="720"></p>
<p align='center'><img src="assets/fig00.png" width="720"></p>
<p align='center'><img src="assets/fig00.png" width="720"></p>
<p align='center'><img src="assets/fig00.png" width="720"></p>

<br>

## 피쳐엔지니어링

### 라벨 인코딩 Label encoding
- `EJ` 열의 unique 값 A, B에 대하여 각각 0과 1로 변환하였습니다.
- `Id` 및 `Class` 열을 제외한 나머지 **feature**들은 모두 numeric 값 입니다.

<br>

### 결측치 처리 Handling Missing Values
**KNNImputer**

- 각 컬럼의 피쳐값들이 정확한 의미를 알 수 없도록 익명화 되어있고, 숫자로 변환되어 있습니다.

- 결측치를 보간하는 방법을 숫자의 특성이 반영될 수 있는 방법을 모색하였습니다.

- KNNImputer는 KNN $^{\textrm{K-Neareast Neighbor}}$ 알고리즘을 사용하여 결측치를 보간함으로 숫자의 거리에 따라 특성이 반영된다고 판단하였습니다.
    - scikit-learn 패키지로 쉽게 사용가능합니다.
    - 결측치 처리에 대한 특별한 해법이 없을 시에도 주로 사용하는 방법입니다.

<br>

### 표준화 Scaler 
**StandardScaler**, **Z-Score**

- 각 `Feature`별 분산과 데이터 값의 범위가 다양하여, 범주형 데이터 `EJ` 열만을 제외하고 스케일을 진행했었습니다. **하지만**

- 최종으로 사용하게 될 `LightGBM` 은 로지스틱회귀나 트리 기반 모델인 의사결정나무, 랜덤 포레스트, 그래디언트 부스팅에 속하는 모델이며 이런 모델은 변수의 크기에 민감하지 않으므로 표준화를 수행해줄 필요가 없습니다.

- 그러므로 **표준화 없이** 기존 데이터를 그데로 사용하게 되었습니다.  

<br>

### 클래스 불균형 Class Imbalacne

- 평가지표를 다시한번 더 설명하자면 `Logloss`는 다른 평가지표와 비교했을 때, 우연히 정답을 맞추는 것을 방지위해 틀린 값에 대하여 큰 페널티를 부여합니다 이는 정확성을 필요로 하는 의료데이터를 평가하는 지표이기 때문일 것입니다.

- 하지만, 수집되는 Class 데이터는 어쩔수 없이 불균형한 데이터가 모이게 될것으로 이런 데이터를 학습시에는 결국 더 많은 수량의 Class 로 예측하게 되는 경향을 보정하기 위해 `Balanced logloss` 최종적으로 리더보드에서 채택하여 평가하고 있습니다.

- 이미 평가지표에서 클래스 불균형을 보정한다고 판단하고 있기에 별도의 처리과정을 진행하지 않았습니다.

<br>

## 모델선정 및 훈련

```python
for bag in range(bag_num):

    print(f'-------------------------- bag: {bag} --------------------------')

    kf = StratifiedKFold(n_splits=n_fold,
                         random_state=42*bag,
                         shuffle=True,
                         )

    models[bag] = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df, greeks['Alpha'])):

        train_df = df.iloc[train_idx]
        valid_df = df.iloc[test_idx]
        valid_ids = valid_df.Id.values.tolist()

        X_train, y_train = train_df[feas_cols], train_df['Class']
        X_valid, y_valid = valid_df[feas_cols], valid_df['Class']

        lgb = LGBMClassifier(**lgbm_params, verbose=-1)

        lgb.fit(X_train, y_train,
                eval_set=(X_valid, y_valid),
                eval_metric=lgb_metric,
                )

        feature_importances = lgb.feature_importances_

        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns, 
            'Split': feature_importances, 
            'Gain': lgb.booster_.feature_importance(importance_type='gain')})
        feature_importance_df['bag'] = bag
        feature_importance_df['fold'] = fold

        feature_importance_df_total = pd.concat([feature_importance_df_total, 
                                                 feature_importance_df], 
                                                 axis=0)

        y_pred = lgb.predict_proba(X_valid)

        logloss = log_loss(y_valid, y_pred)
        balanced_logloss = balanced_log_loss(y_valid, y_pred[:, 1])
        log_losses.append(logloss)
        balanced_log_losses.append(balanced_logloss)

        models[bag].append(lgb)

        print(f"Bags: {bag},
        Fold: {fold}, 
        log loss: {round(logloss, 3)},
        balanced los loss: {round(balanced_logloss, 3)}")
```
```
-------------------------- bag: 0 --------------------------
Bags: 0, Fold: 0, log loss: 0.194, balanced los loss: 0.159
Bags: 0, Fold: 1, log loss: 0.103, balanced los loss: 0.064
Bags: 0, Fold: 2, log loss: 0.241, balanced los loss: 0.158
Bags: 0, Fold: 3, log loss: 0.241, balanced los loss: 0.137
Bags: 0, Fold: 4, log loss: 0.241, balanced los loss: 0.12
-------------------------- bag: 1 --------------------------
Bags: 1, Fold: 0, log loss: 0.277, balanced los loss: 0.177
Bags: 1, Fold: 1, log loss: 0.235, balanced los loss: 0.144
Bags: 1, Fold: 2, log loss: 0.067, balanced los loss: 0.041
Bags: 1, Fold: 3, log loss: 0.233, balanced los loss: 0.119
Bags: 1, Fold: 4, log loss: 0.204, balanced los loss: 0.127
-------------------------- bag: 2 --------------------------
Bags: 2, Fold: 0, log loss: 0.221, balanced los loss: 0.134
Bags: 2, Fold: 1, log loss: 0.161, balanced los loss: 0.106
Bags: 2, Fold: 2, log loss: 0.134, balanced los loss: 0.099
Bags: 2, Fold: 3, log loss: 0.187, balanced los loss: 0.138
Bags: 2, Fold: 4, log loss: 0.212, balanced los loss: 0.106
...
...
...
-------------------------- bag: 18 --------------------------
Bags: 18, Fold: 0, log loss: 0.137, balanced los loss: 0.088
Bags: 18, Fold: 1, log loss: 0.269, balanced los loss: 0.171
Bags: 18, Fold: 2, log loss: 0.122, balanced los loss: 0.075
Bags: 18, Fold: 3, log loss: 0.205, balanced los loss: 0.156
Bags: 18, Fold: 4, log loss: 0.129, balanced los loss: 0.064
-------------------------- bag: 19 --------------------------
Bags: 19, Fold: 0, log loss: 0.291, balanced los loss: 0.155
Bags: 19, Fold: 1, log loss: 0.272, balanced los loss: 0.134
Bags: 19, Fold: 2, log loss: 0.162, balanced los loss: 0.134
Bags: 19, Fold: 3, log loss: 0.136, balanced los loss: 0.069
Bags: 19, Fold: 4, log loss: 0.165, balanced los loss: 0.112
```

- 모델의 성능을 향상시키기 위해 `Bagging`과 `K-Fold Cross Validation`을 사용하여 `LightGBM`을 훈련시켰습니다.

- `Bagging`은 랜덤하게 샘플을 선택하여 여러 모델을 훈련시키고, 그 결과를 집계하는 방법입니다. 이렇게 하면 **모델의 분산을 줄이고 과적합을 방지**할 수 있습니다.

- `Stratified K-Fold Cross Validation`은 데이터를 n_fold 만큼의 그룹으로 나누고, 각 그룹을 테스트 셋으로 사용하는 동시에 나머지 그룹으로 훈련 셋을 구성하는 방식입니다. <br> Stratified는 **각 Fold에 클래스의 비율이 원본 데이터셋의 클래스 비율과 동일하게 유지**됨을 보장하여 **모델의 성능을 일반화해 개선**하도록 합니다. 

- `LightGBM`을 사용했습니다. **빠른 학습 속도**와 **높은 효율성**으로 많이 사용됩니다. CatBoost 사용하지 않은 이유.

- `feature_importances_` 매서드를 통해서 각 특성의 중요도를 가져옵니다. 이는 모델이 예측을 만드는 데 얼마나 각 특성을 사용했는지를 나타냅니다.

- 모델의 성능을 평가합니다. 몇 차례 언급했듯이 `logloss`은 분류 모델의 성능을 평가하는 데 사용되는 손실 함수로, 예측 확률이 실제 레이블과 얼마나 잘 일치하는지를 측정합니다. 

- 또한, `balanced logloss`는 **클래스 불균형을 다루는 방법**으로 사용됩니다.


<br>

## 결과

- 총 6430명 중 513등을 달성했으며 **Top 8%** 이내로 들어와 **첫 Kaggle Competition**에서 **Bronze 메달**을 얻을 수 있었습니다.

- 제가 달성한 평가지표 Balanced logloss는 `private`: `0.42377`,  `public`: `0.17695` 이며,

- 대회 최고 평가지표 Balanced logloss는 `private`: `0.30626`,  `public`: `0.13540` 이었습니다.

<br>

## 한계점 및 회고

- 첫 캐글의 서툼

- greeks 데이터 활용의 부재

- 다양한 시도의 부족 
