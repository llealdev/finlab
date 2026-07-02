from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "financial"
    dense_model: str = "intfloat/multilingual-e5-large"
    sparse_model: str = "Qdrant/bm25"
    colbert_model: str = "colbert-ir/colbertv2.0"
    base_url_api_llm: str = "https://opencode.ai/zen/v1/"
    llm_api_key: str
    llm_model: str = "mimo-v2.5-free"


settings = Settings()
