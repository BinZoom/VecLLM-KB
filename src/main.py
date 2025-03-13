from fastapi import FastAPI
from api.router import router
import uvicorn

app = FastAPI(title="Knowledge Base API")

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)