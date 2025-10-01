# Japanese MapleStory Community Monitoring Automation Script

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/) [![Selenium](https://img.shields.io/badge/Selenium-4-green?style=for-the-badge&logo=selenium)](https://www.selenium.dev/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)](https://huggingface.co/models) [![Notion API](https://img.shields.io/badge/Notion%20API-v1-black?style=for-the-badge&logo=notion)](https://developers.notion.com/)

**A fully automated data pipeline that monitors Japanese gaming communities (5ch, したらば), performs 5-level sentiment analysis using AI, extracts key topics, and updates results to a Notion dashboard in real-time.**

## Purpose

- Monitor raw user feedback from **5ch MapleStory threads** and **したらば exposure threads**
- Track real money trading (RMT) and malicious user activities
- Analyze keywords to identify feedback, balance patch requests, and general opinions
- Identify posting patterns and user trends

<br>

## 🌟 Live Demo

**[➡️ View the live Notion analytics dashboard here](https://vlage.notion.site/26dc0c2b3ce780b5b934e8d25a387c9c?v=26dc0c2b3ce7804c86fa000c1c0bfb13)**

<br>

## 🚀 Features

- **Intelligent Thread Discovery:** Automatically locates the latest forum threads by analyzing search results and listing pages instead of using fixed URLs.
- **Multi-Site Concurrent Crawling:** Reliably collects data from dynamic web pages on 5ch and Shitaraba using `Selenium`.
- **AI-Powered 5-Level Sentiment Analysis:** Uses Hugging Face’s Japanese-optimized BERT model to classify posts into **Very Positive/Positive/Neutral/Negative/Very Negative** plus exception handling.
- **Dynamic Keyword Extraction:** Tags posts with relevant topics based on a keyword list managed via `keywords.xlsx`.
- **Real-Time Notion Dashboard:** Integrates with **Notion API** to automatically update a database with all collection and analysis results for visualization and team sharing.

<br>

## 🏛️ Architecture

`[5ch Search/Shitaraba Lists]` **→** `[Selenium Crawler]` **→** `[SQLite DB]` **→** `[AI Analysis & Keyword Extraction]` **→** `[Notion API]` **→** `[Live Dashboard]`

<br>

## 🛠️ Tech Stack

|Category     |Technology                                  |
|:------------|:-------------------------------------------|
|**Language** |Python                                      |
|**Crawling** |Selenium, BeautifulSoup4, Webdriver-Manager |
|**AI & Data**|Transformers (Hugging Face), PyTorch, Pandas|
|**Database** |SQLite                                      |
|**API**      |Notion Client                               |
|**Packaging**|PyInstaller *(Optional)*                    |

<br>

## ⚙️ Getting Started

#### **1. Prerequisites**

- Python 3.8 or higher
- **(Recommended) NVIDIA GPU with CUDA support:** Maximizes AI analysis speed. Falls back to CPU if unavailable.
- Prepare `keywords.xlsx` file in the project root (sample available in repository)
- Obtain Notion API token and database ID

#### **2. Installation**

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

#### **3. Configuration**

Update the configuration section at the top of `master_pipeline.py` with your Notion credentials:

```python
# --- Configuration ---
# ... (omitted) ...
NOTION_API_KEY = "secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Your Notion API key
NOTION_DATABASE_ID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Your Notion Database ID
# ... (omitted) ...
```

#### **4. Run Pipeline**

```bash
python master_pipeline.py
```

After completion, `game_community_data_COMBINED.db` will be created and all analysis results will be uploaded to your Notion dashboard.

<br>

## 📁 Project Structure

```
.
├── venv/
├── game_community_data_COMBINED.db  (generated after execution)
├── keywords.xlsx
├── master_pipeline.py
├── requirements.txt
└── README.md
```

<br>

-----

### ⚖️ Disclaimer

- This project is created for educational and portfolio purposes only.
- The crawler is designed to respect `robots.txt` rules of target websites.
- All collected data is publicly accessible information.
- Users must comply with the terms of service of target websites.
- The project creator assumes no responsibility for any legal or ethical issues arising from the use of this code. All responsibility lies with the user.