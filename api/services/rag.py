from openai import OpenAI
from config.prompts import RAG_PROMPT
from config.settings import settings
from models.rag import RAGResponse
from services.search import SearchService


class RAGService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.client = OpenAI(
            base_url=settings.base_url_api_llm, api_key=settings.groq_api_key
        )

    def generate_answer(self, query: str, limit: int = 3):
        search_results = self.search_service.search(query, limit)

        context = "\n\n".join(result.text for result in search_results.results)

        prompt = RAG_PROMPT.format(context=context, query=query)

        response = self.client.responses.create(
            model=settings.groq_model,
            input=prompt,
            temperature=0,
            top_p=1,
        )

        metadata = [
            {
                **result.metadata,
                "score": result.score,
            }
            for result in search_results.results
        ]

        return RAGResponse(
            query=query,
            answer=response.output_text,
            metadata=metadata,
        )
