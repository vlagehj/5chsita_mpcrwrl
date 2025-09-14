# 일본 메이플스토리 커뮤니티 모니터링 자동화 스크립트 For 25'넥토리얼 

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/) [![Selenium](https://img.shields.io/badge/Selenium-4-green?style=for-the-badge&logo=selenium)](https://www.selenium.dev/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)](https://huggingface.co/models) [![Notion API](https://img.shields.io/badge/Notion%20API-v1-black?style=for-the-badge&logo=notion)](https://developers.notion.com/)

**서비스 운영의 주요 업무 중 모니터링을 자동화하여 일본의 주요 커뮤니티(5ch, したらば)에 흩어져 있는 유저 반응을 자동으로 수집하고, AI를 통해 텍스트의 감성을 5단계로 정량 분석하며, 핵심 키워드를 추출하여 최종 결과를 Notion 대시보드에 실시간으로 업데이트하는 완전 자동화 데이터 파이프라인입니다.**

## 작성 목적
- **5ch의 Maplestoryスレ**(스레), したらば의 **채널별晒しスレ**(저격/박제 스레)를 통해 날 것 그대로의 의견을 모니터링
- 메루의 현금 거래/악성 이용 유저 파악 및 트래킹
- 키워드 분석을 통해 기타 건의 사항, 밸런스 패치 요구 등 일반적인 의견 모니터링
- 집중 투고 시기/유저 간 트렌드 파악 등 다양한 용도로 사용

<br>

## 🌟 라이브 데모 (Live Demo)

**[➡️ 본 URL을 통해 실시간 Notion 분석 대시보드 확인이 가능합니다](https://vlage.notion.site/26dc0c2b3ce780b5b934e8d25a387c9c?v=26dc0c2b3ce7804c86fa000c1c0bfb13)**

<br>

## 🚀 주요 기능 (Features)

-   **지능형 스레드 탐색:** 고정 URL이 아닌, 검색 결과 및 목록 페이지를 분석하여 항상 최신 게시판 스레드를 자동으로 찾아냅니다.
-   **다중 사이트 동시 크롤링:** `Selenium`을 활용하여 구조가 다른 5ch와 Shitaraba의 동적 웹 페이지 데이터를 안정적으로 수집합니다.
-   **AI 기반 5단계 감성 분석:** `Hugging Face`의 일본어 특화 BERT 모델을 사용하여, 각 게시글의 뉘앙스를 **Very Positive/Positive/Neutral/Negative/Very Negative**의 5단계 분류 + 분류 불가(exception)로 태깅합니다.
-   **동적 키워드 추출:** `keywords.xlsx` 파일을 통해 관리되는 키워드 목록을 기반으로, 각 게시글의 주요 주제를 동적으로 태깅합니다.
-   **실시간 Notion 대시보드:** **Notion API**와 연동하여, 모든 수집 및 분석 결과를 지정된 데이터베이스에 자동으로 업데이트하여 시각적인 결과 확인 및 팀 공유가 가능합니다.

<br>

## 🏛️ 시스템 아키텍처 (Architecture)

`[5ch 검색/Shitaraba 목록]` **->** `[Selenium 크롤러]` **->** `[SQLite 통합 DB]` **->** `[AI 분석 & 키워드 추출]` **->** `[Notion API]` **->** `[실시간 대시보드]`

<br>

## 🛠️ 사용 기술 (Tech Stack)

| 구분 | 기술 |
| :--- | :--- |
| **Language** | Python |
| **Crawling** | Selenium, BeautifulSoup4, Webdriver-Manager |
| **AI & Data** | Transformers (Hugging Face), PyTorch, Pandas |
| **Database** | SQLite |
| **API** | Notion Client |
| **Packaging** | PyInstaller *(Optional)* |

<br>

## ⚙️ 설치 및 실행 방법 (Getting Started)

#### **1. 사전 준비**
-   Python 3.8 이상이 설치되어 있어야 합니다.
-   분석할 키워드를 담은 `keywords.xlsx` 파일을 프로젝트 루트에 준비합니다. (Github내 동일 파일 사용 가능)
-   Notion API 토큰과 데이터베이스 ID를 발급받습니다.

#### **2. 저장소 복제 및 환경 설정**
```bash
# 1. 프로젝트 파일을 다운로드하거나 복제합니다.
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

# 2. 파이썬 가상 환경을 생성하고 활성화합니다.
python -m venv venv
.\venv\Scripts\activate

# 3. requirements.txt 파일로 모든 필요 라이브러리를 한 번에 설치합니다.
pip install -r requirements.txt
```

#### **3. 설정값 입력**
`master_pipeline.py` 스크립트 상단의 설정 영역에 본인의 `NOTION_API_KEY`와 `NOTION_DATABASE_ID`를 입력합니다.

```python
# --- 통합 설정 ---
# ... (생략) ...
NOTION_API_KEY = "secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 본인의 Notion API 키로 변경
NOTION_DATABASE_ID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 본인의 Notion Database ID로 변경
# ... (생략) ...
```

#### **4. 파이프라인 실행**
```bash
python master_pipeline.py
```
실행이 완료되면, 프로젝트 폴더에 `game_community_data_COMBINED.db` 파일이 생성되고 설정된 Notion 페이지에 모든 분석 결과가 자동으로 업로드됩니다.

<br>

## 📁 폴더 구조 (Project Structure)

```
.
├── venv/
├── game_community_data_COMBINED.db  (실행 후 생성)
├── keywords.xlsx
├── master_pipeline.py
├── requirements.txt
└── README.md
```

<br>

---

### ⚖️ 법적 고지 및 윤리적 고려사항 (Disclaimer)

-   본 프로젝트는 오직 교육 및 포트폴리오 목적으로만 제작되었습니다.
-   본 크롤러는 대상 웹사이트의 `robots.txt`에 명시된 규칙을 준수하도록 설계되었습니다.
-   본 프로젝트를 통해 수집되는 데이터는 모두 공개적으로 접근 가능한 정보입니다.
-   본 코드의 사용자는 대상 웹사이트의 서비스 이용 약관을 존중해야 할 의무가 있습니다.
-   본 프로젝트의 코드를 사용하여 발생하는 모든 법적, 윤리적 문제에 대한 책임은 전적으로 사용자 본인에게 있으며, 프로젝트 제작자는 어떠한 책임도 지지 않습니다.
