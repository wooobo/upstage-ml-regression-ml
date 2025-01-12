import pandas as pd

current_version = 'v30'
origin_train_path = '../data/train_origin.csv'
origin_test_path = '../data/test_origin.csv'

local_train_path = '../data/local_train.csv'
local_test_path = '../data/local_test.csv'
local_test_y_path = '../data/local_test_y.csv'

sub_feature_path = '../data/subway_feature.csv'
bus_feature_path = '../data/bus_feature.csv'



def load_local_data():
    train = pd.read_csv(local_train_path, low_memory=False)
    test = pd.read_csv(local_test_path, low_memory=False)
    test_y = pd.read_csv(local_test_y_path, low_memory=False)
    train['is_test'] = 0
    test['is_test'] = 1
    return  pd.concat([train, test]), test_y

def local_data_current_version():
    train = pd.read_csv(local_train_path, low_memory=False)
    test = pd.read_csv(local_test_path, low_memory=False)
    train['is_test'] = 0
    test['is_test'] = 1
    return pd.concat([train, test])

def load_merged_current_version():
    # local_train_v30
    train = loadVersionOfData(f'train_{current_version}')
    test = loadVersionOfData(f'test_{current_version}')

    train['is_test'] = 0
    test['is_test'] = 1
    return pd.concat([train, test], axis=0)


def load_current_version():
    # local_train_v30
    train = loadVersionOfData(f'local_train_{current_version}')
    test = loadVersionOfData(f'local_test_{current_version}')
    test_y = pd.read_csv(f'../data/local_test_y_{current_version}.csv', low_memory=False)

    train['is_test'] = 0
    test['is_test'] = 1
    return pd.concat([train, test], axis=0), test_y

def loadVersionOfData(version):
    return pd.read_csv(f'../data/{version}.csv', low_memory=False)

def load_origin_train():
    return pd.read_csv(origin_train_path, low_memory=False)

def load_origin_test():
    return pd.read_csv(origin_test_path, low_memory=False)

def load_origin_data():
    train = load_origin_train()
    test = load_origin_test()

    train['is_test'] = 0
    test['is_test'] = 1
    return pd.concat([train, test], axis=0)

def load_지하철():
    return pd.read_csv(sub_feature_path, low_memory=False)

def load_버스():
    return pd.read_csv(bus_feature_path, low_memory=False)

def load_아파트_key_xy좌표():
    return pd.read_csv('../data/아파트_unique_key_위도경도.csv', low_memory=False)

def load_주변정보():
    return pd.read_csv('../data/주변시설_apt.csv', low_memory=False)

def load_단지정보():
    return pd.read_

# def load_origin_index_data():
#     train = pd.read_csv(origin_train_index_path, low_memory=False)
#     test = pd.read_csv(origin_test_index_path, low_memory=False)
#
#     train['is_test'] = 0
#     test['is_test'] = 1
#     return pd.concat([train, test], axis=0)
#
#
# def unique_apt_group_file():
#     return pd.read_csv(unique_apt_group_file_name, low_memory=False)
#
# def load_apt_xy_file_path():
#     return pd.read_csv(apt_xy_file_path, low_memory=False)
#
# def 금리_file():
#     return pd.read_csv(금리_file_name, low_memory=False)
#

#
# def load_서울시_단지정보():
#     return pd.read_csv(서울시_단지정보_path, low_memory=False)
#
# def load_아파트_unique_단지정보():
#     return pd.read_csv('../data/result_아파트_단지정보_00_결측치_처리.csv', low_memory=False)
#
# def load_아파트_key_xy좌표():
#     return pd.read_csv('../data/unique_apt_group_xy_v4_최종.csv', low_memory=False)
#
# def load_아파트_key_xy좌표2():
#     return pd.read_csv('../data/아파트_unique_key_위도경도.csv', low_memory=False)
#
# def load_서울시_공동주택_단지현황():
#     return pd.read_csv('../data/지수/서울시 아파트(공동주택) 단지 현황.csv', low_memory=False)
#
# def load_서울시_도시공원_현황():
#     return pd.read_csv('../data//지수/서울시 도시공원 현황.csv', low_memory=False)
#csv('../data/아파트단지_정보_30.csv', low_memory=False)