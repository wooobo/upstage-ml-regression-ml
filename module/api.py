import requests

# @link:https://developers.kakao.com/docs/latest/ko/rest-api/reference#rest-api-list-local
def get_coordinates(address):
    address = address.strip()
    url = "https://dapi.kakao.com/v2/local/search/address.json?query=" + address

    payload = {}
    headers = {
      'Authorization': 'KakaoAK '
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    # JSON 결과를 파싱하여 'x'와 'y' 좌표 반환
    result = response.json()
    if result['documents']:
        # road_address
        if result['documents'][0]['address'] is None:
            address_info = result['documents'][0]['road_address']
        else:
            address_info = result['documents'][0]['address']

        lon = address_info['x']
        lat = address_info['y']
        return lon, lat
    else:
        print("주소를 찾을 수 없습니다.", address)
        return None, None

def search_apt_query(df):
    return df['시군구'] +  " " + df['번지']

original_mapping = {
    "MT1": "대형마트",
    "CS2": "편의점",
    "PS3": "어린이집, 유치원",
    "SC4": "학교",
    "AC5": "학원",
    "PK6": "주차장",
    "OL7": "주유소, 충전소",
    "SW8": "지하철역",
    "BK9": "은행",
    "CT1": "문화시설",
    "AG2": "중개업소",
    "PO3": "공공기관",
    "AT4": "관광명소",
    "AD5": "숙박",
    "FD6": "음식점",
    "CE7": "카페",
    "HP8": "병원",
    "PM9": "약국"
}

def 장소검색_쿼리(x,y,radius,category_group_code):
    reversed_mapping = {value: key for key, value in original_mapping.items()}
    category_code = reversed_mapping.get(category_group_code, None)
    if category_code is None:
        print("없는 카테고리 코드입니다.", category_group_code)
        return None

    return "x=" + str(x) + "&y=" + str(y) + "&radius=" + str(radius) + "&category_group_code=" + str(category_code)


#https://developers.kakao.com/docs/latest/ko/local/dev-guide#search-by-category
def 장소검색(query):
    url = "https://dapi.kakao.com/v2/local/search/category.json?" + query

    payload = {}
    headers = {
      'Authorization': 'KakaoAK '
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    result = response.json()
    return result.get("meta", {}).get("total_count", 0)
