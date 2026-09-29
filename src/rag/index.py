"""Hybrid (dense + BM25) regulation index on embedded Qdrant.

Dense vectors (bge-small) catch paraphrases ("pit lane shut during a stoppage"); BM25 catches
exact regulatory vocabulary ("dry-weather tyres", "suspended"). Results are fused with
Reciprocal Rank Fusion. Everything runs locally: embedded Qdrant storage, fastembed ONNX models.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from src.rag.chunking import RegChunk

DEFAULT_INDEX_PATH = Path("data/qdrant")
COLLECTION = "sporting_regulations"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
# bge models expect this prefix on queries (not documents) for retrieval.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class RetrievedClause:
    chunk: RegChunk
    score: float


class RegulationIndex:
    def __init__(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        self._client = QdrantClient(path=str(path))
        self._dense = TextEmbedding(DENSE_MODEL)
        self._sparse = SparseTextEmbedding(SPARSE_MODEL)

    def rebuild(self, chunks: list[RegChunk], batch_size: int = 64) -> None:
        if self._client.collection_exists(COLLECTION):
            self._client.delete_collection(COLLECTION)
        dim = self._dense.embedding_size
        self._client.create_collection(
            COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.embedding_text() for c in batch]
            dense = self._dense.embed(texts)
            sparse = self._sparse.embed(texts)
            self._client.upsert(
                COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, c.chunk_id)),
                        vector={
                            "dense": d.tolist(),
                            "bm25": models.SparseVector(
                                indices=s.indices.tolist(), values=s.values.tolist()
                            ),
                        },
                        payload=asdict(c),
                    )
                    for c, d, s in zip(batch, dense, sparse, strict=True)
                ],
            )

    def search(self, query: str, season: int, issue: int, k: int = 5) -> list[RetrievedClause]:
        dense = next(self._dense.embed([_QUERY_PREFIX + query])).tolist()
        sparse = next(self._sparse.query_embed(query))
        scope = models.Filter(
            must=[
                models.FieldCondition(key="season", match=models.MatchValue(value=season)),
                models.FieldCondition(key="issue", match=models.MatchValue(value=issue)),
            ]
        )
        result = self._client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(query=dense, using="dense", filter=scope, limit=k * 4),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                    using="bm25",
                    filter=scope,
                    limit=k * 4,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
        )
        return [RetrievedClause(RegChunk(**p.payload), p.score) for p in result.points]

    def close(self) -> None:
        self._client.close()
