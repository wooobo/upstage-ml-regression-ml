from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
import time
import os
from webdriver_manager.chrome import ChromeDriverManager

# 다운로드 디렉토리 설정
download_dir = os.path.join(os.path.expanduser("~"), "kapt_downloads")

# 다운로드 디렉토리가 존재하지 않으면 생성
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--start-maximized")  # 브라우저 창 최대화
# chrome_options.add_argument("--headless")  # 필요시 주석 해제

# 다운로드 관련 설정
prefs = {
    "download.default_directory": download_dir,  # 다운로드 디렉토리 설정
    "download.prompt_for_download": False,       # 다운로드 시 저장 위치 묻지 않음
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True,                # 안전한 다운로드 설정
    "profile.default_content_settings.popups": 0,
    "profile.default_content_setting_values.automatic_downloads": 1,  # 자동 다운로드 허용
}
chrome_options.add_experimental_option("prefs", prefs)

# WebDriver Manager를 사용하여 ChromeDriver 설정
service = Service(ChromeDriverManager().install())

# 웹 드라이버 초기화
driver = webdriver.Chrome(service=service, options=chrome_options)


def wait_for_downloads(download_path, timeout=60):
    """다운로드가 완료될 때까지 대기하는 함수"""
    seconds = 0
    while seconds < timeout:
        # .crdownload 파일이 존재하지 않는지 확인
        if not any([filename.endswith(".crdownload") for filename in os.listdir(download_path)]):
            return True
        time.sleep(3)
        seconds += 3
    return False


try:
    # 웹 페이지 접속
    url = 'https://www.k-apt.go.kr/web/board/webReference/boardList.do'
    driver.get(url)

    # 페이지 로딩 대기
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'headLine')))

    # 고정된 헤더 숨기기 (필요시)
    try:
        driver.execute_script("""
            var header = document.querySelector('.header');
            if(header) {
                header.style.display = 'none';
            }
        """)
        print("고정 헤더 숨김 처리 완료.")
    except Exception as e:
        print("고정 헤더 숨기기 중 오류 발생:", e)

    # 총 페이지 수 파악
    try:
        pagination = driver.find_element(By.CLASS_NAME, 'pagination')
        page_links = pagination.find_elements(By.XPATH, ".//a[@class='page']")
        page_numbers = [int(link.text) for link in page_links if link.text.isdigit()]
        total_pages = max(page_numbers) if page_numbers else 1
        print(f"총 페이지 수: {total_pages}")
    except Exception as e:
        print("총 페이지 수 파악 중 오류 발생:", e)
        total_pages = 1  # 기본값 설정

    # 모든 페이지 순회
    for page in range(2, total_pages + 1):
        try:
            print(f"\n--- 페이지 {page} 처리 시작 ---")

            # 현재 페이지가 이미 선택된 페이지인지 확인
            try:
                current_page_element = driver.find_element(By.XPATH, f"//a[@class='page on' and text()='{page}']")
                if current_page_element:
                    print(f"페이지 {page}가 현재 페이지입니다.")
            except:
                # 페이지 번호 클릭
                page_link = driver.find_element(By.XPATH, f"//a[@class='page' and text()='{page}']")
                if page_link:
                    # 스크롤을 해당 요소로 이동
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", page_link)
                    time.sleep(1)  # 스크롤 후 안정화 대기

                    # ActionChains를 사용하여 요소로 마우스를 이동한 후 클릭
                    actions = ActionChains(driver)
                    actions.move_to_element(page_link).perform()
                    time.sleep(0.5)  # 마우스 이동 후 안정화 대기
                    actions.click(page_link).perform()
                    print(f"페이지 {page}로 이동 완료.")

                    # 페이지 로딩 대기
                    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'headLine')))
                    time.sleep(2)  # 추가 대기
                else:
                    print(f"페이지 {page} 링크를 찾을 수 없습니다.")
                    continue  # 다음 페이지로 넘어감

            # class="headLine"인 모든 요소 찾기
            headlines = driver.find_elements(By.CLASS_NAME, 'headLine')
            print(f"페이지 {page}에 총 {len(headlines)}개의 헤드라인을 찾았습니다.")

            # 각 헤드라인을 순차적으로 클릭하기
            for index in range(len(headlines)):
                try:
                    # 페이지가 동적으로 변경될 수 있으므로, 매번 요소를 다시 찾아야 함
                    headlines = driver.find_elements(By.CLASS_NAME, 'headLine')
                    if index >= len(headlines):
                        print(f"헤드라인 인덱스 {index}이(가) 유효하지 않습니다.")
                        break
                    headline = headlines[index]

                    # 스크롤을 해당 요소로 이동
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", headline)
                    time.sleep(1)  # 스크롤 후 안정화 대기

                    # ActionChains를 사용하여 요소로 마우스를 이동한 후 클릭
                    actions = ActionChains(driver)
                    actions.move_to_element(headline).perform()
                    time.sleep(0.5)  # 마우스 이동 후 안정화 대기
                    actions.click(headline).perform()
                    print(f"페이지 {page} - {index + 1}번째 헤드라인 클릭 완료.")

                    # 상세 페이지 로딩 대기
                    wait.until(EC.element_to_be_clickable((By.ID, 'btn-all-files')))
                    time.sleep(2)  # 추가 대기

                    # 전체 다운로드 버튼 찾기
                    download_button = driver.find_element(By.ID, 'btn-all-files')

                    # JavaScript를 사용하여 다운로드 버튼 클릭
                    driver.execute_script("arguments[0].click();", download_button)
                    print("전체 다운로드 버튼 클릭 완료.")

                    # 다운로드가 완료될 때까지 대기
                    if wait_for_downloads(download_dir):
                        print("다운로드 완료.")
                    else:
                        print("다운로드 완료 확인 시간 초과.")

                    # 이전 페이지로 돌아가기
                    time.sleep(2)  # 추가 대기

                    # driver.back()
                    # print("이전 페이지로 돌아가기.")
                    # "목록" 버튼 찾기
                    list_button = driver.find_element(By.XPATH, "//button[contains(text(), '목록')]")

                    # JavaScript를 사용하여 버튼 클릭 (필요시)
                    driver.execute_script("arguments[0].click();", list_button)
                    print("목록 버튼 클릭 완료.")


                    # 이전 페이지 로딩 대기
                    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'headLine')))
                    time.sleep(2)

                except Exception as e:
                    print(f"페이지 {page} - {index + 1}번째 헤드라인 처리 중 오류 발생:", e)
                    # 오류 발생 시에도 이전 페이지로 돌아가기 시도
                    try:
                        driver.back()
                        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'headLine')))
                        time.sleep(2)
                    except:
                        print("이전 페이지로 돌아가는 중 추가 오류 발생.")
                    continue  # 다음 헤드라인으로 넘어감

            print(f"--- 페이지 {page} 처리 완료 ---")

        except Exception as e:
            print(f"페이지 {page} 처리 중 오류 발생:", e)
            continue  # 다음 페이지로 넘어감

    print("\n모든 페이지의 헤드라인 처리가 완료되었습니다.")

except Exception as e:
    print("오류 발생:", e)
finally:
    # 브라우저 창을 닫지 않고 유지
    print("작업이 완료되었습니다. 브라우저 창을 닫으려면 Enter 키를 누르세요.")
    input("계속하려면 Enter 키를 누르세요.")
    # 드라이버 종료 (원하는 경우 주석 해제)
    # driver.quit()
