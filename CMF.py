import streamlit as st
import pdfplumber
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# =============================
# STREAMLIT CONFIG
# =============================
st.set_page_config(
    page_title="AI Fact Checker",
    layout="wide"
)

st.title("🕵️ AI Fact-Checking Web App")
st.write("Upload a PDF to verify factual claims using live web data.")

# =============================
# ENVIRONMENT CHECK
# =============================
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please set it in environment variables.")
    st.stop()

if not os.getenv("TAVILY_API_KEY"):
    st.error("❌ TAVILY_API_KEY not found. Please set it in environment variables.")
    st.stop()

# =============================
# INITIALIZE MODELS
# =============================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

search_tool = TavilySearchResults(
    max_results=3
)

# =============================
# FUNCTIONS
# =============================
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
    Claims must contain numbers, dates, statistics, or measurable facts.

    Return each claim on a new line.

    TEXT:
    {text}
    """

    response = llm([HumanMessage(content=prompt)])
    lines = response.content.split("\n")

    claims = [
        line.strip("-•1234567890. ")
        for line in lines
        if re.search(r"\d", line)
    ]

    return list(dict.fromkeys(claims))  # remove duplicates


def verify_claim(claim):
    search_results = search_tool.run(claim)

    verification_prompt = f"""
    Claim: {claim}

    Search Results:
    {search_results}

    Decide whether the claim is:
    - Verified
    - Inaccurate
    - False

    Respond in this format:

    Status: <Verified/Inaccurate/False>
    Explanation: <1–2 lines explanation>
    """

    response = llm([HumanMessage(content=verification_prompt)])
    return response.content, search_results


# =============================
# UI
# =============================
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

    st.subheader("🔍 Verification Results")

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
