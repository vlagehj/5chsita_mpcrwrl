# ===================================================================
#  커뮤니티 반응 통합 분석 자동화 파이프라인 (5ch + Shitaraba)
# ===================================================================

import sqlite3, time, re, pandas as pd, datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from notion_client import Client

# --- 통합 설정 ---
FIVCH_SEARCH_URL = "https://ff5ch.syoboi.jp/?q=Maple+Story"
SHITARABA_SUBJECT_URL = "https://jbbs.shitaraba.net/bbs/subject.cgi/netgame/14987/"
SHITARABA_TARGET_KEYWORDS = ["かえで晒しスレ", "ゆかり晒しスレ", "くるみ晒しスレ"]

DB_NAME = 'game_community_data_COMBINED.db' # ★★★ 새로운 통합 DB 이름 ★★★
GAME_TITLE = 'MapleStory'
MODEL_NAME = "koheiduck/bert-japanese-finetuned-sentiment"
NOTION_API_KEY = "secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # ★★★ 본인의 Notion 통합 API 키로 교체 ★★★
NOTION_DATABASE_ID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # ★★★ 본인의 Notion 데이터베이스 ID로 교체 ★★★
KEYWORD_FILE = 'keywords.xlsx'

# --- 함수 정의 ---

def load_keywords_from_excel(file_path):
    # ... (이전과 동일)
    print(f"'{file_path}'에서 키워드 목록을 불러옵니다...")
    try:
        df = pd.read_excel(file_path, header=0, dtype=str)
        keywords = df.iloc[:, 0].dropna().tolist()
        print(f"총 {len(keywords)}개의 키워드를 불러왔습니다.")
        return keywords
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return []

# ★★★ 5ch 크롤링 전용 모듈 (자동 URL 탐색 기능 탑재) ★★★
def crawl_5ch(driver, conn):
    print("\n--- [5ch 크롤링 모듈 시작] ---")
    cur = conn.cursor()
    
    # 1. 5ch 검색 사이트에서 최신 스레드 URL 탐색
    print(f"스레드 검색 페이지({FIVCH_SEARCH_URL})에 접속하여 최신 스레드를 탐색합니다...")
    try:
        driver.get(FIVCH_SEARCH_URL)
        wait = WebDriverWait(driver, 20)
        
        # 스크린샷을 기반으로, 가장 첫 번째 'a.thread' 링크가 최신 스레드
        latest_thread_link = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.thread")))
        fivch_url_to_crawl = latest_thread_link.get_attribute('href')
        print(f"최신 스레드 URL 발견: {fivch_url_to_crawl}")
        
    except Exception as e:
        print(f"5ch 최신 스레드 URL을 찾는 데 실패했습니다: {e}")
        return # URL을 못 찾으면 5ch 크롤링 중단

    # 2. 발견한 URL로 이동하여 데이터 수집
    driver.get(fivch_url_to_crawl)
    wait = WebDriverWait(driver, 60)
    all_button = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "全部")))
    all_button.click()
    wait.until(EC.presence_of_element_located((By.ID, "1")))
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    all_posts_divs = soup.select('div.post')
    print(f"총 {len(all_posts_divs)}개의 게시글을 수집했습니다...")
    
    for post_div in all_posts_divs:
        post_id = post_div.get('id', 'N/A')
        content = post_div.select_one('div.post-content').get_text(strip=True) if post_div.select_one('div.post-content') else ""
        written_time = post_div.select_one('span.date').get_text(strip=True) if post_div.select_one('span.date') else "N/A"
        
        # post_id가 'N/A'가 아닐 경우에만 저장 (광고 등 예외 처리)
        if post_id != 'N/A':
            cur.execute(
                "INSERT OR IGNORE INTO posts (source_site, game_title, post_id, content, source_url, written_time) VALUES (?, ?, ?, ?, ?, ?)",
                ('5ch', GAME_TITLE, f"5ch-{post_id}", content, fivch_url_to_crawl, written_time)
            )
            
    conn.commit()
    print("--- [5ch 크롤링 모듈 완료] ---")


# ★★★ 시타라바 크롤링 전용 모듈 ★★★
# ★★★ 시타라바 크롤링 전용 모듈 (ID 추출 버그 수정) ★★★
def crawl_shitaraba(driver, conn):
    print("\n--- [시타라바 크롤링 모듈 시작] ---")
    cur = conn.cursor()
    
    # 1. 대상 스레드 URL 목록 가져오기
    print(f"스레드 목록 페이지({SHITARABA_SUBJECT_URL}) 접속...")
    driver.get(SHITARABA_SUBJECT_URL)
    time.sleep(3) # 페이지가 자바스크립트를 로드할 시간을 줍니다.
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    thread_links = soup.select('ul.thread-list a[href*="/bbs/read.cgi/"]')
    
    found_threads = {}
    for link in thread_links:
        for keyword in SHITARABA_TARGET_KEYWORDS:
            if keyword in link.get_text(strip=True) and keyword not in found_threads:
                found_threads[keyword] = link.get('href')
                print(f"  -> '{link.get_text(strip=True)}' 스레드 발견: {link.get('href')}")
                break
    
    thread_urls_to_crawl = list(found_threads.values())
    print(f"총 {len(thread_urls_to_crawl)}개의 대상 스레드를 크롤링합니다.")

    # 2. 각 스레드 순회하며 크롤링
    for url in thread_urls_to_crawl:
        print(f"\n  스레드 크롤링 시작: {url}")
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        try:
            all_button = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "全部")))
            all_button.click()
            time.sleep(3) # '全部' 클릭 후 모든 데이터가 로드될 시간을 줍니다.
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            dts = soup.select('dl#thread-body > dt')
            dds = soup.select('dl#thread-body > dd')
            
            thread_id = url.split('/')[-2] # URL에서 스레드 ID 추출
            print(f"  총 {len(dts)}개의 게시글 메타데이터 발견...")
            
            for i in range(1, min(len(dts), len(dds))): # 1번 원글은 제외
                dt, dd = dts[i], dds[i]
                
                # ★★★ 핵심 수정: 이제 <b> 태그가 아닌 <dt> 태그의 id 속성에서 게시글 번호를 직접 가져옵니다. ★★★
                post_num = dt.get('id', 'comment_N/A').replace('comment_', '')
                unique_post_id = f"stb-{thread_id}-{post_num}"
                
                content = dd.get_text(strip=True)
                
                time_match = re.search(r'(\d{4}/\d{2}/\d{2}\(.\) \d{2}:\d{2}:\d{2})', dt.get_text())
                written_time = time_match.group(1) if time_match else "N/A"
                
                cur.execute(
                    "INSERT OR IGNORE INTO posts (source_site, game_title, post_id, content, source_url, written_time) VALUES (?, ?, ?, ?, ?, ?)",
                    ('Shitaraba', GAME_TITLE, unique_post_id, content, url, written_time)
                )
        except Exception as e:
            print(f"  스레드 '{url}' 크롤링 중 오류: {e}")
    conn.commit()
    print("--- [시타라바 크롤링 모듈 완료] ---")
    
# ★★★ 통합 데이터 수집 함수 (Orchestrator) ★★★
def run_crawling_and_storage():
    print("="*30 + "\n1단계: 통합 데이터 수집 및 저장을 시작합니다.\n" + "="*30)
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_site TEXT, game_title TEXT,
            post_id TEXT UNIQUE, content TEXT, sentiment TEXT,
            sentiment_score REAL, keywords TEXT, source_url TEXT,
            written_time TEXT
        )
    ''')
    conn.commit()
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    try:
        crawl_5ch(driver, conn)
        crawl_shitaraba(driver, conn)
        print("\n모든 사이트의 데이터 수집이 완료되었습니다.")
    finally:
        driver.quit()
        conn.close()

def run_sentiment_analysis():
    """AI 감성 분석"""
    print("\n" + "="*30 + "\n2단계: AI 감성 분석을 시작합니다.\n" + "="*30)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row 
    cur = conn.cursor()
    cur.execute("SELECT id, content FROM posts WHERE sentiment IS NULL")
    rows_to_analyze = cur.fetchall()
    print(f"총 {len(rows_to_analyze)}개의 신규 게시글에 대한 감성 분석을 시작합니다.")
    for row in rows_to_analyze:
        post_id, content = row['id'], row['content']
        final_sentiment, final_score = 'Exception', 0.0
        if content and len(content) > 5:
            try:
                ai_result = sentiment_analyzer(content[:512])[0]
                label = ai_result['label'].lower()
                score = ai_result['score']
                final_score = score
                if label == 'positive':
                    if score > 0.95: final_sentiment = 'Very Positive'
                    else: final_sentiment = 'Positive'
                elif label == 'negative':
                    if score > 0.95: final_sentiment = 'Very Negative'
                    else: final_sentiment = 'Negative'
                else:
                    final_sentiment = 'Neutral'
            except Exception: pass
        cur.execute("UPDATE posts SET sentiment = ?, sentiment_score = ? WHERE id = ?", (final_sentiment, final_score, post_id))
    conn.commit()
    conn.close()
    print("2단계 완료!")

def run_keyword_extraction(keywords_list):
    """키워드 추출"""
    print("\n" + "="*30 + "\n3단계: 고유 명사 키워드 추출을 시작합니다.\n" + "="*30)
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row 
    cur = conn.cursor()
    cur.execute("SELECT id, content FROM posts WHERE keywords IS NULL")
    rows_to_analyze = cur.fetchall()
    print(f"총 {len(rows_to_analyze)}개의 신규 게시글에 대한 키워드 분석을 시작합니다.")
    for row in rows_to_analyze:
        post_id, content = row['id'], row['content']
        if not content: continue
        found_keywords = {keyword for keyword in keywords_list if keyword in content}
        if found_keywords:
            cur.execute("UPDATE posts SET keywords = ? WHERE id = ?", (",".join(found_keywords), post_id))
    conn.commit()
    conn.close()
    print("3단계 완료!")

# ★★★ Notion 업데이트 함수 (Source 속성 추가) ★★★
def update_notion_database():
    print("\n" + "="*30 + "\n4단계: Notion 데이터베이스 업데이트를 시작합니다.\n" + "="*30)
    notion = Client(auth=NOTION_API_KEY)
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row 
    cur = conn.cursor()

    print("Notion에서 기존 데이터 ID를 조회합니다...")
    try:
        existing_pages = notion.databases.query(database_id=NOTION_DATABASE_ID)["results"]
        existing_post_ids = {page['properties']['Post_ID']['title'][0]['text']['content'] for page in existing_pages}
        print(f"총 {len(existing_post_ids)}개의 기존 데이터가 Notion에 존재합니다.")
    except Exception as e:
        print(f"Notion 조회 실패: {e}")
        conn.close()
        return

    cur.execute("SELECT * FROM posts")
    all_posts = cur.fetchall()
    new_posts = [post for post in all_posts if str(post['post_id']) not in existing_post_ids]
    
    print(f"총 {len(new_posts)}개의 신규 데이터를 Notion에 업로드합니다...")
    
    # ★★★ 1. 연속 실패 카운터 및 한계치 설정 ★★★
    consecutive_failures = 0
    FAILURE_LIMIT = 5

    for post in new_posts:
        # ... (page_properties 만드는 부분은 이전과 동일)
        page_properties = {
            "Post_ID": {"title": [{"text": {"content": str(post['post_id'])}}]},
            "Sentiment": {"select": {"name": str(post['sentiment'])}},
            "Score": {"number": round(post['sentiment_score'], 4)},
            "URL": {"url": str(post['source_url'])},
            "Content": {"rich_text": [{"text": {"content": str(post['content'])[:2000]}}]},
            "Source": {"select": {"name": str(post['source_site'])}}
        }
        if post['keywords']:
            page_properties["Keywords"] = {"multi_select": [{"name": kw} for kw in str(post['keywords']).split(',')]}
        raw_date_str = post['written_time']
        if raw_date_str and raw_date_str != "N/A":
            try:
                cleaned_date_str = re.sub(r'\([月火水木金土日]\)', '', raw_date_str)
                try:
                    dt_object = datetime.datetime.strptime(cleaned_date_str.strip(), "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    dt_object = datetime.datetime.strptime(cleaned_date_str.strip(), "%Y/%m/%d %H:%M:%S")
                page_properties["Written_Time"] = {"date": {"start": dt_object.isoformat()}}
            except ValueError: pass

        try:
            notion.pages.create(parent={"database_id": NOTION_DATABASE_ID}, properties=page_properties)
            # ★★★ 2. 업로드 성공 시, 카운터를 0으로 초기화 ★★★
            consecutive_failures = 0
            
        except Exception as e:
            print(f"[ID: {post['post_id']}] 업로드 실패: {e}")
            # ★★★ 3. 업로드 실패 시, 카운터 1 증가 ★★★
            consecutive_failures += 1
            print(f"    -> 연속 실패: {consecutive_failures}/{FAILURE_LIMIT}")

        # ★★★ 4. 카운터가 한계치에 도달했는지 확인 ★★★
        if consecutive_failures >= FAILURE_LIMIT:
            print(f"\n오류: Notion 업로드가 {FAILURE_LIMIT}회 연속 실패하여 작업을 중단합니다.")
            print("    API 키, 데이터베이스 ID, 인터넷 연결 또는 Notion 서버 상태를 확인해주세요.")
            break  # for 루프를 탈출하여 함수를 안전하게 종료

    conn.close()
    print("4단계 완료!")

# --- 메인 파이프라인 실행 ---
if __name__ == "__main__":
    start_time = time.time()
    
    # 0단계: 엑셀에서 키워드 불러오기
    maple_keywords = load_keywords_from_excel(KEYWORD_FILE)
    
    run_crawling_and_storage()
    run_sentiment_analysis()
    run_keyword_extraction(maple_keywords)
    update_notion_database()
    
    end_time = time.time()
    print(f"\n모든 파이프라인이 성공적으로 완료되었습니다. (총 소요 시간: {end_time - start_time:.2f}초)")