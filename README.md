# AI Fact-Checking Web App

A deployed web application that automatically extracts factual claims from a PDF
and verifies them against **live web data**.

The app is designed to be **robust, production-safe**, and resilient to real-world
API limitations such as quota exhaustion.

---

## 🚀 Live Demo
🔗 **Deployed App URL:**  
(Add your Streamlit URL here)

---

## 🎯 Objective

This application acts as a **fact-checking layer** between content drafts and publishing.
It identifies factual claims in documents and verifies their accuracy using
real-time web sources.

---

## 🧠 How It Works

### 1️⃣ Claim Extraction
- The app extracts factual claims (numbers, dates, statistics) from an uploaded PDF.
- Primary method: OpenAI LLM
- Fallback method: Regex-based extraction (used if LLM quota is exhausted)

### 2️⃣ Live Verification
- Each extracted claim is verified using **Tavily Web Search API**
- The verification logic analyzes live search results to determine whether a claim is:
  - **Verified**
  - **Inaccurate**
  - **False**

### 3️⃣ Reporting
- Each claim is displayed along with:
  - Verification status
  - Short explanation
  - Live web sources used

---

## 🛡️ API Quota Handling (Important)

This app is intentionally designed to **degrade gracefully** under API limitations.

### 🔹 If OpenAI quota is available:
- LLM-based claim extraction is used
- Tavily is used for live verification

### 🔹 If OpenAI quota is exhausted:
- The app automatically switches to **regex-based claim extraction**
- **Live web verification continues using Tavily**
- The app does **not crash** and remains fully testable

This ensures uninterrupted evaluation and reflects real-world production reliability.

---

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **LLM:** OpenAI (gpt-3.5-turbo)
- **Web Search:** Tavily API
- **Frameworks:** LangChain
- **PDF Parsing:** pdfplumber

---

## 📂 Project Structure
fact-checker-app/
│
├── CMF.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml

