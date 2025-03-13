from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from typing import List, Union
from config.config import settings
import os


class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def load_document(self, file_path: str) -> List[str]:
        _, ext = os.path.splitext(file_path)

        # Document Loading
        if ext.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')

        documents = loader.load()
        # Document splitting
        texts = self.text_splitter.split_documents(documents)
        return [doc.page_content for doc in texts]
