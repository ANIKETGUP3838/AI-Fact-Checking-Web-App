import streamlit as st
import pdfplumber
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="AI Fact Checker", layout="wide")
st.title("🕵️ AI Fact-Checking Web App")
st.write("Upload a PDF to verify factual claims using live web data.")

# =====================================================
# LOAD API KEYS (MANDATORY)
# =====================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing. Live verification cannot run.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY missing. Live verification cannot run.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# =====================================================
# INITIALIZE MODELS (LIVE ONLY)
# =====================================================
try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",   # universally available
        temperature=0
    )
    search_tool = TavilySearchResults(max_results=3)
except Exception as e:
    st.error("❌ Failed to initialize OpenAI or Tavily")
    st.exception(e)
    st.stop()

# =====================================================
# FUNCTIONS
# =====================================================
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_claims(text):
    prompt = f"""
    Extract ONLY factual, verifiable claims from the text below.
    Claims must include numbers, dates, statistics, or measurable facts.
    Return each claim on a new line.

    TEXT:
    {text}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        st.error("❌ OpenAI claim extraction failed")
        st.exception(e)
        st.stop()

    lines = response.content.split("\n")
    claims = [
        line.strip("-• ")
        for line in lines
        if re.search(r"\d", line)
    ]

    return list(dict.fromkeys(claims))


def verify_claim(claim):
    try:
        search_results = search_tool.run(claim)
    except Exception as e:
        st.error("❌ Tavily web search failed")
        st.exception(e)
        st.stop()

    verification_prompt = f"""
    Claim: {claim}

    Web Search Results:
    {search_results}

    Classify the claim as:
    - Verified
    - Inaccurate
    - False

    Respond in this format:
    Status: <Verified/Inaccurate/False>
    Explanation: <1–2 lines explanation>
    """

    try:
        response = llm.invoke([HumanMessage(content=verification_prompt)])
        return response.content, search_results
    except Exception as e:
        st.error("❌ OpenAI verification failed")
        st.exception(e)
        st.stop()

# =====================================================
# UI
# =====================================================
uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    with st.spinner("🧠 Extracting factual claims..."):
        claims = extract_claims(text)

    if not claims:
        st.warning("No factual claims detected in the document.")
        st.stop()

    st.subheader("📌 Extracted Claims")
    for i, claim in enumerate(claims, 1):
        st.markdown(f"**{i}. {claim}**")

    st.subheader("🔍 Live Verification Results")

    for i, claim in enumerate(claims, 1):
        with st.spinner(f"Verifying claim {i}..."):
            verdict, sources = verify_claim(claim)

        st.markdown("---")
        st.markdown(f"### Claim {i}")
        st.markdown(f"**{claim}**")
        st.markdown(verdict)

        st.markdown("**Sources:**")
        st.json(sources)
