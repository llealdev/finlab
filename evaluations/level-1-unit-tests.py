import instructor
import json
import os

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url="https://opencode.ai/zen/v1/", api_key=os.getenv("LLM_API_KEY")
)
client = instructor.from_openai(client, mode=instructor.Mode.JSON)


def load_test_case(filename: str) -> str:
    with open(f"test_cases/{filename}", "r") as f:
        return json.load(f)


COMPANY_TICKER_MAPPINGS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "tesla": "TSLA",
    "meta": "META",
}

TICKER_EXTRACTION_PROMPT = """You are a stock ticker symbol extractor. Given a user message, extract the stock ticker symbol for any publicly traded company mentioned.

Rules:
- Return ONLY the ticker symbol (e.g., AAPL, TSLA, MSFT)
- If no company is mentioned, return "NONE"
- If multiple companies are mentioned, return the first/main one
- Use standard US stock exchange ticker symbols (NYSE, NASDAQ)

Examples:
User: "How is Disney doing?" → DIS
User: "What about Tesla's performance?" → TSLA
User: "Tell me about IBM" → IBM
User: "What's the weather today?" → NONE

User message: {query}
Response:"""


class TickerResult(BaseModel):
    ticker: str = Field(
        pattern="^[A-Z]{1,5}$",
        description="Stock ticker symbol (1-5 uppercase letters)",
    )


def extract_ticker(query: str) -> str | None:
    query_lower = query.lower()
    for company_name, ticker in COMPANY_TICKER_MAPPINGS.items():
        if company_name in query_lower:
            return ticker

    prompt = TICKER_EXTRACTION_PROMPT.format(query=query)

    response = client.chat.completions.create(
        model="deepseek-v4-flash-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_model=TickerResult,
    )
    ticker = response.ticker

    return ticker if ticker != "NONE" else None


def test_static_mapping_apple():
    test_case = load_test_case("apple_test.json")
    results = extract_ticker(test_case["query"])
    assert results == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']}, got {results}"
    )


def test_llm_fallback_ibm():
    test_case = load_test_case("ibm_test.json")
    results = extract_ticker(test_case["query"])
    assert results == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']}, got {results}"
    )


def test_no_company_mentioned():
    test_case = load_test_case("no_company_test.json")
    results = extract_ticker(test_case["query"])
    assert results == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']}, got {results}"
    )


def test_natural_language_query():
    test_case = load_test_case("natural_language_test.json")
    results = extract_ticker(test_case["query"])
    assert results == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']}, got {results}"
    )


if __name__ == "__main__":
    tests = [
        test_static_mapping_apple,
        test_llm_fallback_ibm,
        test_no_company_mentioned,
        test_natural_language_query,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1

        except AssertionError as e:
            print(f"FAIL {test.__name__}: {e}")

    print(f"\n Results: {passed}/{len(tests)} tests passed")
