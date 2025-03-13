from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Milvus Config
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "knowledge_base"

    # LLM Config
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4o"

    # Document segmentation configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # Vector dimension
    VECTOR_DIM: int = 768

    # LangSmith
    LANGSMITH_API_KEY: str = "false"
    LANGSMITH_TRACING: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
