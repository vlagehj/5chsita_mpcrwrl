# ===================================================================
#  커뮤니티 반응 통합 분석 자동화 파이프라인 (5ch + Shitaraba)
# ===================================================================

import sqlite3, time, re, pandas as pd, datetime
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from notion_client import Client

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("="*50)
print(f"사용할 연산 장치: {DEVICE.upper()}")
if DEVICE == 'cpu':
    print("경고: CUDA 지원 GPU가 감지되지 않았습니다. AI 분석이 CPU로 진행됩니다.")
print("="*50)

# --- 통합 설정 ---
FIVCH_SEARCH_URL = "https://ff5ch.syoboi.jp/?q=Maple+Story"
SHITARABA_SUBJECT_URL = "https://jbbs.shitaraba.net/bbs/subject.cgi/netgame/14987/"
SHITARABA_TARGET_KEYWORDS = ["かえで晒しスレ", "ゆかり晒しスレ", "くるみ晒しスレ"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DB_NAME = 'game_community_data_COMBINED.db' # ★★★ 통합 DB 이름 
DB_PATH = os.path.join(SCRIPT_DIR, DB_NAME)
KEYWORD_FILE_PATH = os.path.join(SCRIPT_DIR, 'keywords.xlsx')
GAME_TITLE = 'MapleStory'
MODEL_NAME = "koheiduck/bert-japanese-finetuned-sentiment"
NOTION_API_KEY = "secret_xxxxxxxxxxxxxxxxxxxxx"   # ★★★ 본인의 Notion 통합 API 키로 교체 ★★★
NOTION_DATABASE_ID = "secret_xxxxxxxxxxxxxxxxxxxxx" # ★★★ 본인의 Notion 데이터베이스 ID로 교체 ★★★
KEYWORD_FILE = 'keywords.xlsx'

# --- 함수 정의 ---

def load_keywords_from_excel(file_path):

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
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_site TEXT, game_title TEXT,
            post_id TEXT UNIQUE, content TEXT, 
            content_kr TEXT,
            sentiment TEXT, sentiment_score REAL, 
            keywords TEXT, source_url TEXT,
            written_time TEXT
        )
    ''')
    conn.commit()
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
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
    device_index = 0 if DEVICE == 'cuda' else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    conn = sqlite3.connect(DB_PATH)
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

def run_keyword_extraction(keywords_list):
    """키워드 추출"""
    print("\n" + "="*30 + "\n3단계: 고유 명사 키워드 추출을 시작합니다.\n" + "="*30)
    conn = sqlite3.connect(DB_PATH)
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

def run_translation():
    print("\n" + "="*30 + "\n4단계: AI 번역을 시작합니다. (게시글 수에 따라 매우 오래 걸릴 수 있습니다)\n" + "="*30)
        
    TRANSLATION_MODEL = "trillionlabs/Tri-1.8B-Translation"
        
    print(f"'{TRANSLATION_MODEL}' 생성형 번역 모델을 로딩합니다...")
        
    try:
        # ★★★ device_map="auto"를 사용하여 accelerate가 자동으로 장치를 할당하도록 합니다. ★★★
           tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
           model = AutoModelForCausalLM.from_pretrained(TRANSLATION_MODEL, device_map="auto", dtype=torch.bfloat16)
           print("모델 로딩 완료!")
    except Exception as e:
            print(f"번역 모델 로딩 실패: {e}")
            return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row 
    cur = conn.cursor()
    cur.execute("SELECT id, content FROM posts WHERE content_kr IS NULL AND content IS NOT NULL")
    rows_to_translate = cur.fetchall()
    
    print(f"총 {len(rows_to_translate)}개의 신규 게시글에 대한 번역을 시작합니다.")
    
    for row in rows_to_translate:
        post_id, content = row['id'], row['content']
        truncated_content = content[:300] # 입력 길이를 적절히 조절
        
        ## ★★★ 프롬프트 설계 ★★★ << 이 프로젝트의 핵심입니다. 프롬프트를 입력하여 커스터마이징이 가능합니다.
        prompt = f"""Translate the following Japanese MapleStory Game Community user's text into Korean:{truncated_content} <ko>"""
        
        messages = [
            {"role": "system", "content": "You are an expert translator specializing in the gaming community, especially MapleStory. You are fluent in the slang and nuances of both Japanese and Korean gamers. Your task is to translate Japanese text into natural, modern Korean that a Korean gamer would actually use."},
            {"role": "user", "content": prompt}
        ]

        try:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # 토크나이저를 사용하여 모델이 이해할 수 있는 입력값으로 변환합니다.
            inputs = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt",
                add_generation_prompt=True
            ).to(model.device)

            # 모델을 통해 새로운 텍스트를 생성(번역)합니다.
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 공식 문서의 방법대로, 프롬프트를 제외한 순수 번역 결과만 정확하게 추출합니다.
            translated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

            cur.execute("UPDATE posts SET content_kr = ? WHERE id = ?", (translated_text, post_id))
            print(f"[ID: {post_id}] 번역 완료.")
        except Exception as e:
            print(f"[ID: {post_id}] 번역 중 오류 발생: {e}")

    conn.commit()
    conn.close()
    print("4단계 완료!")

# ★★★ Notion 업데이트 함수  ★★★
def update_notion_database():
    print("\n" + "="*30 + "\n4단계: Notion 데이터베이스 업데이트를 시작합니다.\n" + "="*30)
    notion = Client(auth=NOTION_API_KEY)
    conn = sqlite3.connect(DB_PATH)
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
        
        page_properties = {
            "Post_ID": {"title": [{"text": {"content": str(post['post_id'])}}]},
            "Sentiment": {"select": {"name": str(post['sentiment'])}},
            "Score": {"number": round(post['sentiment_score'], 4)},
            "URL": {"url": str(post['source_url'])},
            "Content": {"rich_text": [{"text": {"content": str(post['content'])[:2000]}}]},
            "Content_KR": {"rich_text": [{"text": {"content": str(post['content_kr'])[:2000]}}]},        
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
    print("5단계 완료!")

# --- 메인 파이프라인 실행 ---
if __name__ == "__main__":
    start_time = time.time()
    
    # 0단계: 엑셀에서 키워드 불러오기
    maple_keywords = load_keywords_from_excel(KEYWORD_FILE)
    
    # 1단계: 5ch + 시타라바 크롤링 및 DB 저장
    run_crawling_and_storage()
    
    # 2단계: AI 감성 분석
    run_sentiment_analysis()

    # 3단계: 키워드 추출
    run_keyword_extraction(maple_keywords)

    # 4단계: AI KOJA 번역
    run_translation()

    # 5단계: Notion 업데이트
    update_notion_database()
    
    end_time = time.time()
    print(f"\n모든 파이프라인이 성공적으로 완료되었습니다. (총 소요 시간: {end_time - start_time:.2f}초)")