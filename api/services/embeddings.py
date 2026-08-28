from config.settings import settings
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding


class EmbeddingService:
    def __init__(self):
        self.dense_model = TextEmbedding(settings.dense_model)
        self.sparse_model = SparseTextEmbedding(settings.sparse_model)
        self.colbert_model = LateInteractionTextEmbedding(settings.colbert_model)

    def embe_query(self, query: str):
        dense = list(self.dense_model.query_embed([query]))[0].tolist()
        sparse = list(self.sparse_model.query_embed([query]))[0].as_object()
        colbert = list(self.colbert_model.query_embed([query]))[0].tolist()
        return dense, sparse, colbert
