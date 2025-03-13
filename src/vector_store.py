from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from config.config import settings


class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self._connect_milvus()
        self._init_collection()
        self.vectorstore = Milvus(
            embedding_function=self.embeddings,
            collection_name=settings.MILVUS_COLLECTION,
            connection_args={"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT},
            auto_id=True
        )

    def _connect_milvus(self):
        """connect Milvus Server"""
        connections.connect(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )

    def _init_collection(self):
        """init Milvus collection"""
        if utility.has_collection(settings.MILVUS_COLLECTION):
            self.collection = Collection(settings.MILVUS_COLLECTION)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=settings.VECTOR_DIM)
        ]
        schema = CollectionSchema(fields=fields, description="Vector storage of knowledge base documents")
        self.collection = Collection(settings.MILVUS_COLLECTION, schema)

        # Specify the index of the vector
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index("vector", index_params)

    def add_texts(self, texts):
        """Add text to vector storage"""
        self.vectorstore.add_texts(texts)

    def similarity_search(self, query: str, k: int = 4):
        """similarity search"""
        return self.vectorstore.similarity_search(query, k=k)
