import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import prepare_query, retrieve, stream_answer, public_sources

app = FastAPI(title="LegalRAG")

# Tighten allow_origins to your real frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = []


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    history = [t.model_dump() for t in req.history][-6:]

    def event_stream():
        try:
            prep = prepare_query(req.message, history)
            hits = retrieve(prep["search_query"], req.message, prep["doc_type"])

            # 1) sources first, so the UI can render citations immediately
            yield sse("sources", public_sources(hits))

            # 2) answer tokens
            for token in stream_answer(req.message, hits, prep["language"], history):
                yield sse("token", token)

            yield sse("done", {"language": prep["language"]})
        except Exception as e:
            print(f"[chat] error: {e}")
            yield sse("error", "Something went wrong. Please try again.")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
