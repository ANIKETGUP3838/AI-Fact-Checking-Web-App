import streamlit as st
import pdfplumber
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import RateLimitError

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="AI Fact Checker", layout="wide")
st.title("🕵️ AI Fact-Checking Web App")
st.write("Upload a PDF to verify factual claims using live web data.")

# =====================================================
# LOAD API KEYS
# =====================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))

if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY missing. Live verification cannot run.")
    st.stop()

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

OPENAI_AVAILABLE = True
if not OPENAI_API_KEY:
    OPENAI_AVAILABLE = False
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =====================================================
# INITIALIZE TOOLS
# =====================================================
search_tool = TavilySearchResults(max_results=5)

llm = None
if OPENAI_AVAILABLE:
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    except Exception:
        OPENAI_AVAILABLE = False

# =====================================================
# HELPERS
# =====================================================
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def regex_claim_extraction(text):
    lines = text.split("\n")
    claims = [l.strip() for l in lines if re.search(r"\d", l)]
    return claims[:5]


# =====================================================
# CLAIM EXTRACTION
# =====================================================
def extract_claims(text):
    global OPENAI_AVAILABLE

    if not OPENAI_AVAILABLE:
        return regex_claim_extraction(text)

    prompt = f"""
    Extract ONLY factual, verifiable claims from the text below.
    Claims must include numbers, dates, or statistics.
    Return each claim on a new line.

    TEXT:
    {text}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        lines = response.content.split("\n")
        claims = [l.strip("-• ") for l in lines if re.search(r"\d", l)]
        return list(dict.fromkeys(claims))

    except RateLimitError:
        OPENAI_AVAILABLE = False
        return regex_claim_extraction(text)

    except Exception:
        OPENAI_AVAILABLE = False
        return regex_claim_extraction(text)


# =====================================================
# TAVILY-ONLY VERIFICATION (NO OPENAI)
# =====================================================
def verify_claim(claim):
    try:
        results = search_tool.run(claim)
    except Exception as e:
        return (
            "Status: Unknown\n"
            "Explanation: Web search failed."
        ), None

    text_blob = str(results).lower()

    # Simple heuristic verification
    if any(word in text_blob for word in ["according", "reported", "data", "statistics"]):
        status = "Verified"
    elif any(word in text_blob for word in ["myth", "false", "incorrect", "debunked"]):
        status = "False"
    else:
        status = "Inaccurate"

    explanation = (
        f"Status: {status}\n"
        "Explanation: Determined using live web search results."
    )

    return explanation, results


# =====================================================
# UI
# =====================================================
uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    with st.spinner("🧠 Extracting claims..."):
        claims = extract_claims(text)

    if not claims:
        st.warning("No factual claims detected.")
        st.stop()

    st.subheader("📌 Extracted Claims")
    for i, claim in enumerate(claims, 1):
        st.markdown(f"**{i}. {claim}**")

    st.subheader("🔍 Live Verification Results (Web-based)")

    for i, claim in enumerate(claims, 1):
        with st.spinner(f"Verifying claim {i}..."):
            verdict, sources = verify_claim(claim)

        st.markdown("---")
        st.markdown(f"### Claim {i}")
        st.markdown(f"**{claim}**")
        st.markdown(verdict)

        if sources:
            st.markdown("**Sources:**")
            st.json(sources)
