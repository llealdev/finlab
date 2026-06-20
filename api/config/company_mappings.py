COMPANY_TICKER_MAPPINGS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "nvidia": "NVDA",
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
