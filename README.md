# AI Fact-Checking Web App

A deployed web application that automatically extracts factual claims from a PDF
and verifies them against **live web data**.

This app is designed to be **robust, production-safe**, and resilient to real-world
API limitations such as OpenAI quota exhaustion.

---

## 🚀 Live Demo
🔗 **Deployed App URL:**  
(Add your Streamlit app link here)

---

## 🎯 Objective

The goal of this project is to build a **fact-checking layer** that sits between
content drafts and publishing. The system identifies factual claims and checks
their accuracy using **real-time web search**.

---

## 🧠 How It Works

### 1️⃣ Claim Extraction
- The app extracts factual claims (numbers, dates, statistics) from an uploaded PDF.
- **Primary method:** OpenAI LLM (when available)
- **Fallback method:** Regex-based extraction when LLM quota is exhausted

### 2️⃣ Live Verification
- Claims are verified using **Tavily Web Search API**
- The app analyzes live search results to classify each claim as:
  - **Verified**
  - **Inaccurate**
  - **False**

### 3️⃣ Reporting
For every claim, the app displays:
- Verification status
- Short explanation
- Live web sources used

---

## 🛡️ API Quota Handling (Important)

This app is intentionally built to **degrade gracefully** under API limitations.

### If OpenAI quota is available:
- LLM-based claim extraction is used
- Tavily is used for live verification

### If OpenAI quota is exhausted:
- The app switches to **regex-based claim extraction**
- **Live web verification continues using Tavily**
- The app does **not crash** and remains fully testable

This design reflects real-world production reliability and ensures uninterrupted evaluation.

---

## 🧰 Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **LLM:** OpenAI (`gpt-3.5-turbo`)  
- **Web Search:** Tavily API  
- **Framework:** LangChain  
- **PDF Parsing:** pdfplumber  

---

## 📂 Project Structure
```text
fact-checker-app/
│
├── CMF.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

