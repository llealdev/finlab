import asyncio
from openai import AsyncOpenAI
import instructor
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
from models.agent import (
    AgentResponse,
    FinalRecommendation,
    FundamentalAnalysis,
    MomentumAnalysis,
    SentimentAnalysis,
)

from services.search import SearchService


class AgentServices:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        client = AsyncOpenAI(
            base_url=settings.base_url_api_llm, api_key=settings.llm_api_key
        )
        self.client = instructor.from_openai(client=client, model=instructor.Mode.JSON)

    def _run_queries(self, queries: list[str], limit: int, filter: dict = None):
        all_results = []
        for query in queries:
            search_result = self.search_service.search(
                query=query, limit=limit, filter=filter
            )
            all_results.extend([result.text for result in search_result.results])
        return "\n\n".join(all_results)

    async def _generate_completion(self, prompt: str, response_model=None):
        return await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_model=response_model,
        )

    async def _analyze_fundamental(self, ticker: str, limit: int):
        filter = {"ticker": ticker, "form_type": "10-K"}
        context = self._run_queries(FUNDAMENTAL_QUERIES, limit=limit, filter=filter)
        prompt = FUNDAMENTAL_PROMPT.format(context=context)
        return await self._generate_completion(
            prompt=prompt, response_model=FundamentalAnalysis
        )

    async def _analyze_momentum(self, ticker: str, limit: int):
        filter = {"ticker": ticker, "form_type": "10-Q"}
        context = self._run_queries(MOMENTUM_QUERIES, limit=limit, filter=filter)
        prompt = MOMENTUM_PROMPT.format(context=context)
        return await self._generate_completion(
            prompt=prompt, response_model=MomentumAnalysis
        )

    async def _analyze_sentiment(self, ticker: str, limit: int):
        filter = {"ticker": ticker, "source": "yahoo_finance"}
        query = SENTIMENT_QUERY_TEMPLATE.format(ticker=ticker)
        results = self.search_service.search(query=query, limit=limit, filter=filter)
        context = "\n\n".join([result.text for result in results.results])
        prompt = SENTIMENT_PROMPT.format(context=context)
        return await self._generate_completion(
            prompt=prompt, response_model=SentimentAnalysis
        )

    async def analyze(self, ticker: str, limit: int = 3):
        fundamental_task = self._analyze_fundamental(ticker=ticker, limit=limit)
        momentum_task = self._analyze_momentum(ticker=ticker, limit=limit)
        sentiment_task = self._analyze_sentiment(ticker=ticker, limit=limit)

        (
            fundamental_analysis,
            momentum_analysis,
            sentiment_analysis,
        ) = await asyncio.gather(fundamental_task, momentum_task, sentiment_task)

        aggregation_prompt = AGGREGATION_PROMPT.format(
            fundamental=fundamental_analysis.model_dump_json(indent=2),
            momentum=momentum_analysis.model_dump_json(indent=2),
            sentiment=sentiment_analysis.model_dump_json(indent=2),
        )

        final_recomendation = await self._generate_completion(
            aggregation_prompt, FinalRecommendation
        )

        return AgentResponse(
            ticker=ticker,
            fundamental_analysis=fundamental_analysis,
            momentum_analysis=momentum_analysis,
            sentiment_analysis=sentiment_analysis,
            final_recommendation=final_recomendation,
        )
