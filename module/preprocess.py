from unittest.mock import inplace

from module import file_load
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import TimeSeriesSplit

한강위도경도 = [
    [126.826622, 37.588015],
    [126.861538, 37.569796],
    [126.885039, 37.555692],
    [126.890859, 37.552130],
    [126.904283, 37.543953],
    [126.925775, 37.538105],
    [126.945825, 37.526956],
    [126.958155, 37.517086],
    [126.981778, 37.510733],
    [126.996010, 37.515852],
    [127.012776, 37.527276],
    [127.021246, 37.535866],
    [127.034671, 37.536734],
    [127.057256, 37.529926],
    [127.064574, 37.526408],
    [127.091596, 37.524397],
    [127.098798, 37.528647],
    [127.103869, 37.533902],
    [127.112453, 37.542810],
    [127.113779, 37.544592],
    [127.118791, 37.556058],
    [127.132043, 37.569074],
    [127.149098, 37.572499],
    [127.161716, 37.577660],
    [126.833011, 37.585040, ],
    [126.843883, 37.579665, ],
    [126.851064, 37.575001, ],
    [126.857647, 37.572156, ],
    [126.863532, 37.567808, ],
    [126.871212, 37.564171, ],
    [126.879689, 37.558716, ],
    [126.896346, 37.548911, ],
    [126.915528, 37.538989, ],
    [126.931146, 37.536230, ],
    [126.950571, 37.521202, ],
    [126.964626, 37.514814, ],
    [126.969995, 37.513185, ],
    [126.989577, 37.511870, ],
    [127.005133, 37.521139, ],
    [127.000000, 37.518070, ],
    [127.008923, 37.524396, ],
    [127.017608, 37.531409, ],
    [127.025741, 37.538359, ],
    [127.039717, 37.534978, ],
    [127.048561, 37.533288, ],
    [127.073434, 37.523581, ],
    [127.085673, 37.522705, ],
    [127.095780, 37.526337, ],
    [127.107308, 37.538171, ],
    [127.116309, 37.551694, ],
    [127.118678, 37.556514, ],
    [127.121916, 37.563087, ],
    [127.127917, 37.565841, ],
    [127.140708, 37.571035, ],
    [127.148841, 37.571286, ],
    [127.158080, 37.575979, ],
    [127.165976, 37.578733, ],
    [126.985364, 37.505763],
    [126.988733, 37.507282],
    [126.994092, 37.509346],
    [127.000522, 37.512383],
    [127.002896, 37.513537],
    [127.007718, 37.516026],
    [127.009250, 37.517909],
    [127.011852, 37.520823],
    [127.013919, 37.523434],
    [127.017747, 37.526166],
    [127.019431, 37.527016],
    [127.021422, 37.528837],
    [127.022876, 37.530052],
    [127.023412, 37.531266],
    [127.025785, 37.533269],
    [127.030761, 37.533876],
    [127.033288, 37.533755],
    [127.038723, 37.532176],
    [127.048369, 37.528959],
    [127.047986, 37.529020],
    [127.048752, 37.529202],
    [127.054034, 37.527441],
    [127.057326, 37.524527],
    [127.061001, 37.522523],
    [127.064752, 37.519730],
    [127.077996, 37.517301],
    [127.082208, 37.517360],
    [127.089133, 37.517774],
    [127.093898, 37.519014],
]

# 이상치 제거 방법에는 IQR을 이용하겠습니다.
def remove_outliers_iqr(dt, column_name):
    df = dt.query('is_test == 0')  # train data 내에 있는 이상치만 제거하도록 하겠습니다.
    df_test = dt.query('is_test == 1')

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(lower_bound)
    print(upper_bound)
    df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    result = pd.concat([df, df_test])  # test data와 다시 합쳐주겠습니다.
    return result


def 아파트_unique_key_add(df):
    df.loc[df['아파트명'] == '서초포레스타2단지', '번지'] = df.loc[df['아파트명'] == '서초포레스타2단지', '번지'].fillna('384')
    df.loc[df['아파트명'] == '힐스테이트 서초 젠트리스', '번지'] = df.loc[df['아파트명'] == '힐스테이트 서초 젠트리스', '번지'].fillna('557')
    df['아파트명'] = df['아파트명'].fillna(df['도로명'])

    assert df['아파트명'].isnull().sum() == 0, "아파트명에 결측치가 있습니다."
    df['아파트_unique_key'] = df['구'] + "_" + df['동'] + "_" + df['번지'].astype(str) + "_" + df['아파트명'] + "_" + df[
        '건축년도'].astype(str)


def 시_군_구_add(df):
    df[['시', '구', '동']] = df['시군구'].str.split(' ', expand=True)
    del df['시군구']


def 날짜포맷_add(df):
    df["계약연도"] = df["계약년월"].astype(str).str[:4]  # 첫 4글자는 연도
    df["계약월"] = df["계약년월"].astype(str).str[4:]  # 나머지 글자는 월

    df['계약일'] = df['계약일'].apply(lambda x: f'{x:02d}')

    df["계약년월일"] = df["계약년월"].astype(str) + df["계약일"].astype(str)


def to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])


def target이상치제거(df_target: pd.DataFrame):
    print('target이상치제거')
    df_target['계약일'] = df_target['계약일'].apply(lambda x: f'{x:02d}')
    df_target["계약년월일"] = df_target["계약년월"].astype(str) + df_target["계약일"].astype(str)

    df_target_ths = df_target.query('is_test == 0')
    df_test = df_target.query('is_test == 1')

    # 계약년월일을 datetime 형식으로 변환
    df_target_ths['계약년월일_dt'] = pd.to_datetime(df_target_ths['계약년월일'])
    # 그룹화 및 정렬
    sorted_df = df_target_ths.sort_values(by=['시군구', '번지', '아파트명', '평수', '계약년월일_dt'])
    # 이전 가격 추가
    sorted_df['이전가격'] = sorted_df.groupby(['시군구', '번지', '아파트명', '평수'])['target'].shift(1)
    # 하락률 계산
    sorted_df['하락률'] = ((sorted_df['target'] - sorted_df['이전가격']) / sorted_df['이전가격']) * 100

    # 35% 이상 하락한 경우 제거
    cond = (sorted_df['이전가격'].notnull()) & (sorted_df['하락률'] <= -35)

    sorted_df = sorted_df[~cond]
    sorted_df.drop(['계약년월일_dt', '이전가격', '하락률','계약년월일'], axis=1, inplace=True)
    return pd.concat([sorted_df, df_test])


def 하락률_filter(df):
    # 계약년월일을 datetime 형식으로 변환
    df_target_ths['계약년월일_dt'] = pd.to_datetime(df_target_ths['계약년월일'])
    # 그룹화 및 정렬
    sorted_df = df_target_ths.sort_values(by=['시군구', '번지', '아파트명', '평수', '계약년월일_dt'])
    # 이전 가격 추가
    sorted_df['이전가격'] = sorted_df.groupby(['시군구', '번지', '아파트명', '평수'])['target'].shift(1)
    # 하락률 계산
    sorted_df['하락률'] = ((sorted_df['target'] - sorted_df['이전가격']) / sorted_df['이전가격']) * 100

    # 35% 이상 하락한 경우 제거
    cond = (sorted_df['이전가격'].notnull()) & (sorted_df['하락률'] <= -35)

    sorted_df = sorted_df[~cond]
    sorted_df.drop(['계약년월일_dt', '이전가격', '하락률'], axis=1, inplace=True)


# KDTree를 사용하여 가장 가까운 지하철 거리 계산
def calculate_nearest_subway_distance(real_estate_df, taget_dt, target_name='nearest_subway_distance'):
    taget_dt = taget_dt.rename(columns={'위도': 'y', '경도': 'x'})

    # 지하철 위치로 KDTree 생성
    subway_tree = KDTree(taget_dt[['x', 'y']].values)

    # 부동산 위치에 대해 가장 가까운 지하철 거리 계산
    distances, _ = subway_tree.query(real_estate_df[['좌표X', '좌표Y']].values)

    # 거리 정보를 새로운 컬럼으로 추가
    real_estate_df[target_name] = distances
    real_estate_df[target_name] = real_estate_df[target_name].apply(lambda x: x * 100000)
    return real_estate_df


def calculate_nearest_bus_distance(real_estate_df, taget_dt, target_name='nearest_bus_distance'):
    taget_dt = taget_dt.rename(columns={'Y좌표': 'y', 'X좌표': 'x'})

    # 버스 위치로 KDTree 생성
    bus_tree = KDTree(taget_dt[['x', 'y']].values)

    # 부동산 위치에 대해 가장 가까운 버스 거리 계산
    distances, _ = bus_tree.query(real_estate_df[['좌표X', '좌표Y']].values)

    # 거리 정보를 새로운 컬럼으로 추가
    real_estate_df[target_name] = distances
    real_estate_df[target_name] = real_estate_df[target_name].apply(lambda x: x * 100000)
    return real_estate_df


def nearest_subway_name(real_estate_df, taget_dt, target_name):
    # 지하철 위치로 KDTree 생성
    subway_tree = KDTree(taget_dt[['x', 'y']].values)

    # 부동산 위치에 대해 가장 가까운 지하철 거리 계산
    _, indices = subway_tree.query(real_estate_df[['좌표X', '좌표Y']].values)
    # indices를 1D 배열로 변환
    indices = indices.flatten()

    # 거리 정보를 새로운 컬럼으로 추가
    real_estate_df[target_name] = taget_dt.iloc[indices]['호선'].values
    return real_estate_df


def convert_m2_to_pyong(area_m2):
    return int(area_m2 / 3.3058)  # Truncate decimal places


def 평수_add(df):
    df.rename(columns={'전용면적(㎡)': '전용면적'}, inplace=True)
    df["평수"] = df["전용면적"].apply(convert_m2_to_pyong)


def 아파트주변정보(df):
    주변정보_df = file_load.load_주변정보()
    주변정보_filter = 주변정보_df[['아파트_unique_key', '편의점', '학원', '학교', '공공기관', '병원', '음식점', '어린이집, 유치원']]

    return df.merge(주변정보_filter, left_on='아파트_unique_key', right_on="아파트_unique_key", how='left')


def rename(df):
    return df.rename(columns={'전용면적(㎡)': '전용면적'})


def 해제사유발생유무_add(df):
    df['해제사유발생유무'] = df['해제사유발생일'].notnull().astype(int)


def 번지_본번_부번_결측채우기(df):
    df['도로명'] = df['도로명'].replace(' ', np.nan)
    df['아파트명'] = df['아파트명'].fillna(df['도로명'])
    df['도로명'] = df['도로명'].fillna(df['아파트명'])
    df['번지'] = df['번지'].fillna(df['도로명'])
    # df['본번'] = df['본번'].fillna(df['번지']).astype(str)
    # df['부번'] = df['부번'].fillna(df['번지']).astype(str)


def 계약년월일(df):
    df['계약년월일'] = df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2)


def 강남_add(df):
    all = list(df['구'].unique())
    gangnam = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
    gangbuk = [x for x in all if x not in gangnam]

    assert len(all) == len(gangnam) + len(gangbuk)

    # 강남의 여부를 체크합니다.
    is_gangnam = []
    for x in df['구'].tolist():
        if x in gangnam:
            is_gangnam.append(1)
        else:
            is_gangnam.append(0)

    # 파생변수를 하나 만릅니다.
    df['강남여부'] = is_gangnam


def 신축_add(df):
    def categorize_building_age(years):
        if years <= 3:
            return '신축'
        # elif 4 <= years <= 5:
        #     return '준축'
        # elif 6 <= years <= 10:
        #     return '중축'
        else:
            return '구축'

    df['신축구분'] = df['계약_건축년도_차이'].apply(categorize_building_age)


def 계약_건축년도_차이(df):
    df['계약_건축년도_차이'] = df["계약년월"].astype(str).str[:4].astype(int) - df['건축년도'].astype(int)


# def 건물연령구분_add(df):
#     df['건물연령구분'] = df["계약년월"].astype(str).str[:4].astype(int) - df['건축년도']


def 탑층_add(df):
    df['temp_최대층'] = df.groupby('아파트_unique_key')['층'].transform('max')

    df['is_top'] = (df['temp_최대층'] == df['층']).astype(int)

    # 임시 열(temp_최대층) 제거
    df.drop(columns=['temp_최대층'], inplace=True)


def 층그룹_add(df):
    # 층수 구간과 레이블 정의
    bins = [-float('inf'), 0, 10, 20, 30, 50, 60, float('inf')]  # 경계값 설정
    labels = ['지하', '10미만', '10이상-20미만', '20이상-30미만', '30이상-50미만', '50이상-60미만', '60이상']  # 레이블 설정

    # 층수 그룹화
    df['층_그룹'] = pd.cut(df['층'], bins=bins, labels=labels, right=False)


def 평수그룹_add(df):
    # 평수 구간 및 레이블 정의
    bins = [-float('inf'), 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
    labels = ['10미만', '10이상-20미만', '20이상-30미만', '30이상-40미만',
              '40이상-50미만', '50이상-60미만', '60이상-70미만',
              '70이상-80미만', '80이상-90미만', '90이상-100미만', '100이상']

    # 평수 그룹화
    df['평수_그룹'] = pd.cut(df['평수'], bins=bins, labels=labels, right=False)


def 생활지수파생변수(df):
    # df['산책로밀도'] = df['산책로총길이'] / df['산책로갯수']   # 22723.45592710044
    df['산책로밀도'] = df.apply(
        lambda row: row['산책로총길이'] / row['산책로갯수'] if row['산책로갯수'] != 0 else 0,
        axis=1
    )
    df.drop(['산책로총길이', '산책로갯수'], axis=1, inplace=True)  # 불필요한 컬럼을 제거합니다.
    # df['생태문화길밀도'] = df['생태문화길총길이'] / df['생태문화길갯수'] # 22754.426097650645
    df['생태문화길밀도'] = df.apply(
        lambda row: row['생태문화길총길이'] / row['생태문화길갯수'] if row['생태문화길갯수'] != 0 else 0,
        axis=1
    )
    df.drop(['생태문화길총길이', '생태문화길갯수'], axis=1, inplace=True)  # 불필요한 컬럼을 제거합니다.


def 테스트_최대평수이상제거(df):
    return df[df['평수'] <= 91]


def 테스트_최대층이상제거(df):
    return df[df['층'] <= 65]


def 건축년도_계약년도_앞서면제거(df):
    return df.query('경과연도 >= 0')


def is_상가_add(df):
    df['is_상가'] = df['아파트명'].str.contains(r'(상가)', regex=True, na=False)


def 한강거리_add(df):
    # 한강 좌표를 KDTree로 구성
    han_river_tree = KDTree(한강위도경도)

    distances, _ = han_river_tree.query(df[['좌표X', '좌표Y']].values)
    df['한강거리'] = distances
    df['한강거리'] = df['한강거리'].apply(lambda x: x * 100000)


def 고급아파트_추가(df):
    # 고급 아파트 키워드 목록
    keywords = [
        '아크로', '힐스테이트', '퍼스트', '퍼스트타워', '퍼스트힐스',
        '래미안 리더스원', '래미안라클래시', '래미안청담로이뷰', '래미안퍼스티지'
                                           '자이', '더샵', '아이파크', '롯데캐슬', '푸르지오', 'e편한세상',
        '센트레빌', '트리마제', '갤러리아포레', '마크힐스', '파크리오',
        '더플래스', '라브르', '더피크', '반포주공1단지', '아크로리버파크', '한강맨숀', '아크로리버뷰신반포',
        '디에이치아너힐즈', '현대14차', '반포래미안아이파크', '도곡렉슬', '현대5차', '현대3차', '신반포2', '한양7'
        , '현대8차', '개포주공'
    ]
    # 아크로서울포레스트,갤러리아포레,한남더힐,반포차이,현차1차,현대2차
    # 키워드를 '|'로 연결하여 정규 표현식 패턴 생성
    pattern = '|'.join(keywords)
    # 아파트명에 패턴이 포함되어 있는지 여부를 '고급아파트' 열에 저장
    df['고급아파트'] = df['아파트명'].str.contains(pattern).astype(int)


def 고급아파트_추가2(df):
    apt = '개포주공4단지', '도곡렉슬', '동부센트레빌', '디에이치아너힐즈', '디팰리스', '래미안 리더스원', '래미안대치팰리스', '래미안라클래시', '래미안신반포팰리스', '래미안신반포팰리스', '래미안청담로이뷰', '래미안퍼스티지', '반포래미안아이파크', '반포센트럴자이', '반포자이', '반포주공1단지', '서초그랑자이', '센트럴파크', '센트럴파크', '신반포2', '신반포자이', '아크로리버뷰신반포', '아크로리버파크', '에이아이디차관주택', '청담린든그로브', '타워팰리스1', '트리마제', '한강맨숀', '한양3', '한양4', '한양5', '한양7', '현대14차', '현대3차', '현대5차', '현대8차'
    # df['고급아파트'] = df['아파트명'].apply(lambda x: 1 if x in apt else 0)
    df['고급아파트'] = df.apply(lambda row: 1 if row['아파트명'] in apt and row['계약년월'] >= 201801 else 0, axis=1)


def 코로나시기(df):
    df['코로나시기'] = df['계약년월'].apply(lambda x: 1 if x > 202006 else 0)


def 비싼아파트_add(df):
    # 아파트_unique_key별 평균 target 계산
    avg_target = df.groupby('아파트_unique_key')['target'].mean().reset_index()
    avg_target.rename(columns={'target': 'avg_target'}, inplace=True)

    # 평균 target 값이 200,000 이상이면 비싼아파트로 표시
    avg_target['비싼아파트'] = (avg_target['avg_target'] >= 300000).astype(int)

    # 원본 데이터와 병합
    df2 = pd.merge(df, avg_target[['아파트_unique_key', '비싼아파트']], on='아파트_unique_key')
    return df2


def 아파트주변정보(df):
    주변정보_df = file_load.load_주변정보()
    주변정보_filter = 주변정보_df[['아파트_unique_key', '편의점', '학원', '학교', '공공기관', '병원', '음식점', '어린이집, 유치원']]

    return df.merge(주변정보_filter, left_on='아파트_unique_key', right_on="아파트_unique_key", how='left')


def log처리(df):
    # 변환할 컬럼 리스트
    columns_to_transform = [
        '경과연도',
        'nearest_park_distance',
        '한강거리',
        'nearest_subway_distance',
        'nearest_bus_distance',
        '공공기관',
        '음식점',
        '병원',
        '학교',
        '학원', '공원밀도'
    ]

    # log1p 변환 적용
    for col in columns_to_transform:
        df[col] = np.log1p(df[col])


def 공원밀도_add(df):
    df['공원밀도'] = df['공원갯수'] / df['공원총면적']
    df['공원밀도'].fillna(0, inplace=True)


def label_encoding(dt_train, dt_test):
    # 파생변수 제작으로 추가된 변수들이 존재하기에, 다시한번 연속형과 범주형 칼럼을 분리해주겠습니다.
    continuous_columns_v2 = []
    categorical_columns_v2 = []

    for column in dt_train.columns:
        if pd.api.types.is_numeric_dtype(dt_train[column]):
            continuous_columns_v2.append(column)
        else:
            categorical_columns_v2.append(column)

    print("연속형 변수:", continuous_columns_v2)
    print("범주형 변수:", categorical_columns_v2)

    label_encoders = {}
    for col in tqdm(categorical_columns_v2):
        lbl = LabelEncoder()
        lbl.fit(dt_train[col].astype(str))
        dt_train[col] = lbl.transform(dt_train[col].astype(str))
        label_encoders[col] = lbl
        for label in np.unique(dt_test[col]):
            if label not in lbl.classes_:  #
                lbl.classes_ = np.append(lbl.classes_, label)  #
        dt_test[col] = lbl.transform(dt_test[col].astype(str))
    assert dt_train.shape[1] == dt_test.shape[1]


def 컬럼_정리(df):
    df.rename(columns={'전용면적(㎡)': '전용면적'}, inplace=True)


def drop_columns(df, drop_columns):
    return df.drop(columns=drop_columns)


def 범주형_filter(df):
    continuous_columns_v2 = []
    categorical_columns_v2 = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continuous_columns_v2.append(column)
        else:
            categorical_columns_v2.append(column)
    return categorical_columns_v2


def 범주형_encoding(dt_train, dt_test, categorical_columns_v2):
    label_encoders = {}

    for col in tqdm(categorical_columns_v2):
        lbl = LabelEncoder()

        lbl.fit(dt_train[col].astype(str))
        dt_train[col] = lbl.transform(dt_train[col].astype(str))
        label_encoders[col] = lbl  # 나중에 후처리를 위해 레이블인코더를 저장해주겠습니다.

        for label in np.unique(dt_test[col]):
            if label not in lbl.classes_:  # unseen label 데이터인 경우
                lbl.classes_ = np.append(lbl.classes_, label)  # 미처리 시 ValueError발생하니 주의하세요!

        dt_test[col] = lbl.transform(dt_test[col].astype(str))

    return label_encoders

def split_train_test(model_data):
    dt_train = model_data.query('is_test == 0')
    dt_test = model_data.query('is_test == 1')

    return dt_train.drop(['is_test'], axis=1), dt_test.drop(['is_test'], axis=1)

# - 위 데이터를 이용해 모델을 train 해보겠습니다. 모델은 RandomForest를 이용하겠습니다.
# - Train과 Valid dataset을 분할하는 과정에서는 `holdout` 방법을 사용하겠습니다. 이 방법의 경우  대략적인 성능을 빠르게 확인할 수 있다는 점에서 baseline에서 사용해보도록 하겠습니다.
#   - 이 후 추가적인 eda를 통해서 평가세트와 경향을 맞추거나 kfold와 같은 분포에 대한 고려를 추가할 수 있습니다.
def holdout_split_X_y(dt_train):
    X = dt_train.drop(['target'], axis=1)
    y = dt_train['target']

    # Hold out split을 사용해 학습 데이터와 검증 데이터를 8:2 비율로 나누겠습니다.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2023)
    return X_train, X_val, y_train, y_val

def holdout_split_X_y_time_series(dt_train):
    y = dt_train['target']
    X = dt_train.drop(['target'], axis=1)
    # Time Series Split 설정 (5개 분할)
    tscv = TimeSeriesSplit(n_splits=5)
    # 마지막 분할을 사용하여 train/val 분리
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    return X_train, X_val, y_train, y_val

def 단지정보_add(df):
    단지정보 = file_load.load_단지정보()[['클러스터','아파트_unique_key']]
    return df.merge(단지정보, on='아파트_unique_key', how='left').reset_index(drop=True)


def holdout_split_X_y_sorted(data, target_column='target', holdout_ratio=0.2):
    """
    데이터를 입력(X)과 레이블(y)로 분리하고, 마지막 holdout_ratio 비율만큼 검증 데이터로 나눕니다.

    Parameters:
        data (pd.DataFrame): 입력 데이터프레임
        target_column (str): 레이블이 있는 컬럼명 (기본값: 'target')
        holdout_ratio (float): 검증 데이터로 사용할 비율 (기본값: 0.1)

    Returns:
        X_train (pd.DataFrame): 학습용 입력 데이터
        X_val (pd.DataFrame): 검증용 입력 데이터
        y_train (pd.Series): 학습용 레이블
        y_val (pd.Series): 검증용 레이블
    """
    # 데이터 정렬 (날짜 기준으로 정렬한다고 가정)
    data = data.sort_values(by='계약년월').reset_index(drop=True)

    # holdout 데이터 크기 계산
    holdout_size = int(len(data) * holdout_ratio)

    # 입력(X)과 레이블(y) 분리
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 검증 데이터와 학습 데이터 분리
    X_train, X_val = X.iloc[:-holdout_size], X.iloc[-holdout_size:]
    y_train, y_val = y.iloc[:-holdout_size], y.iloc[-holdout_size:]

    return X_train, X_val, y_train, y_val


def kfold_split_X_y(data, target_column='target', n_splits=5):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # KFold 객체 생성
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 결과 저장 리스트
    folds = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        folds.append((X_train, X_val, y_train, y_val))

    return folds