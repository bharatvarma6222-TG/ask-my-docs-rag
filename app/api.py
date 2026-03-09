from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.rag import ingest_pdf, answer_query

app = FastAPI(title="Ask My Docs - Hybrid RAG")


@app.get("/")
def home():
    return {"message": "API is running. Go to /docs to use Swagger UI."}


@app.post("/ingest/pdf")
async def ingest_pdf_route(file: UploadFile = File(...)):
    try:
        data = await file.read()
        doc_id = ingest_pdf(data, filename=file.filename)
        return {"status": "ok", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_route(req: QueryRequest):
    try:
        return answer_query(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
