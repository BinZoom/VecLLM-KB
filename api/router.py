from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from src.chat_manager import ChatManager
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
import tempfile
import os

router = APIRouter()
chat_manager = ChatManager()
doc_loader = DocumentLoader()
vector_store = VectorStore()

class QueryRequest(BaseModel):
    thread_id: str
    query: str

class QueryResponse(BaseModel):
    answer: str

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        response = chat_manager.get_response(
            thread_id=request.thread_id,
            query=request.query
        )
        return QueryResponse(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        try:
            # Loading and splitting documents
            texts = doc_loader.load_document(temp_path)
            vector_store.add_texts(texts)
            return {"message": "Document uploaded and processed successfully"}
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear_memory/{thread_id}")
async def clear_memory(thread_id: str):
    try:
        chat_manager.memory_manager.clear_memory(thread_id)
        return {"message": f"Memory cleared for thread {thread_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))