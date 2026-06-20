import os

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()


qdrant = QdrantClient(
    api_key=os.getenv("QDRANT_API_KEY"),
    url=os.getenv("QDRANT_URL"),
)

fields_to_indexs = [
    "source.ticker",
    "source.form_type",
    "source.source",
]

for field_name in fields_to_indexs:
    qdrant.create_payload_index(
        collection_name="financial",
        field_name=field_name,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    print(f"Índice criado para {field_name}")
