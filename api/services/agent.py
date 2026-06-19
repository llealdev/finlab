import asyncio
from openai import AsyncOpenAI
from config.prompts import (
    FUNDAMENTAL_QUERIES,
    MOMENTUM_QUERIES,
    SENTIMENT_QUERY_TEMPLATE,
    FUNDAMENTAL_PROMPT,
    MOMENTUM_PROMPT,
    SENTIMENT_PROMPT,
    AGGREGATION_PROMPT,
)
from config.settings import settings
from models.agent import AgentResponse

from services.search import SearchService


class AgentServices:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.client = AsyncOpenAI(
            base_url=settings.base_url_api_llm, api_key=settings.llm_api_key
        )

    def _run_queries(self, queries: list[str], limit: int):
        all_results = []
        for query in queries:
            search_result = self.search_service.search(query=query, limit=limit)
            all_results.extend([result.text for result in search_result.results])
        return "\n\n".join(all_results)

    async def _generate_completion(self, prompt: str):
        response = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        return response.choices[0].message.content

    async def _analyze_fundamental(self, limit: int):
        context = self._run_queries(FUNDAMENTAL_QUERIES, limit=limit)
        prompt = FUNDAMENTAL_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def _analyze_momentum(self, limit: int):
        context = self._run_queries(MOMENTUM_QUERIES, limit=limit)
        prompt = MOMENTUM_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def _analyze_sentiment(self, ticker: str, limit: int):
        query = SENTIMENT_QUERY_TEMPLATE.format(ticker=ticker)
        results = self.search_service.search(query=query, limit=limit)
        context = "\n\n".join([result.text for result in results.results])
        prompt = SENTIMENT_PROMPT.format(context=context)
        return await self._generate_completion(prompt=prompt)

    async def analyze(self, ticker: str, limit: int = 3):
        fundamental_task = self._analyze_fundamental(limit=limit)
        momentum_task = self._analyze_momentum(limit=limit)
        sentiment_task = self._analyze_sentiment(ticker=ticker, limit=limit)

        (
            fundamental_analysis,
            momentum_analysis,
            sentiment_analysis,
        ) = await asyncio.gather(fundamental_task, momentum_task, sentiment_task)

        aggregation_prompt = AGGREGATION_PROMPT.format(
            fundamental=fundamental_analysis,
            momentum=momentum_analysis,
            sentiment=sentiment_analysis,
        )

        final_recomendation = await self._generate_completion(aggregation_prompt)

        return AgentResponse(
            ticker=ticker,
            fundamental_analysis=fundamental_analysis,
            momentum_analysis=momentum_analysis,
            sentiment_analysis=sentiment_analysis,
            final_recommendation=final_recomendation,
        )
