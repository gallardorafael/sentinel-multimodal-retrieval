import os

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "mm_retrieval_collection")
VECTOR_FIELD_DIM = 768
DEFAULT_VECTOR_FIELD_NAME = "embedding"
DEFAULT_METRIC = "COSINE"
DEFAULT_FIELDS = ["filename", "name"]
