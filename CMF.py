import streamlit as st
import pdfplumber
import re
import os

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import HumanMessage

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Fact Checker", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=TAVILY_API_KEY
)

# ======================
# FUNCTIONS
# ======================
def extract_text_from_pdf(pdf):
    text = ""
    with pdfplumber.open(pdf) as p:
        for page in p.pages:
            text += page.extract_text() + "\n"
    return text


def extract_claims(text):
    """
    Uses LLM to extract verifiable claims
    """
    prompt = f"""
    Extract ONLY factual, verifiable claims from the text below.
    Claims must include statistics, numbers, dates, or factual statements.

    Return them as a numbered list.

    TEXT:
    {text}
    """

    response = llm([HumanMessage(content=prompt)])
    claims = response.content.split("\n")

    return [c for c in claims if re.search(r"\d", c)]


def verify_claim(claim):
    """
    Verifies claim using live web search
    """
    search_results = search_tool.run(claim)

    verification_prompt = f"""
    Claim: {claim}

    Web Search Results:
    {search_results}

    Based on the search results, classify the claim as:
    - Verified
    - Inaccurate
    - False

    Give a 1–2 line explanation and mention the correct fact if applicable.
    """

    response = llm([HumanMessage(content=verification_prompt)])

    return response.content, search_results


# ======================
# UI
# ======================
st.title("🕵️ AI Fact-Checking Web App")
st.write("Upload a PDF and automatically verify claims using live web data.")

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Identifying factual claims..."):
        claims = extract_claims(text)

    st.subheader("📌 Extracted Claims")

    if not claims:
        st.warning("No verifiable claims found.")
    else:
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
