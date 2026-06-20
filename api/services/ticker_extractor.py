import instructor
from openai import OpenAI
from config.company_mappings import COMPANY_TICKER_MAPPINGS, TICKER_EXTRACTION_PROMPT
from config.settings import settings
from models.ticker_extractor import TickerResult


class TickerExtractor:
    def __init__(self):
        client = OpenAI(
            base_url=settings.base_url_api_llm, api_key=settings.llm_api_key
        )
        self.client = instructor.from_openai(client, mode=instructor.Mode.JSON)
        self.mappings = COMPANY_TICKER_MAPPINGS

    def _extract_with_llm(self, query: str):
        prompt = TICKER_EXTRACTION_PROMPT.format(query=query)

        result = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_model=TickerResult,
        )

        return result.ticker

    def extract_ticker(self, query: str) -> str | None:
        query_lower = query.lower()
        for company_name, ticker in self.mappings.items():
            if company_name in query_lower:
                return ticker

        return self._extract_with_llm(query)
