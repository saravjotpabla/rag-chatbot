from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from retriever import load_index, retrieve
from chain import build_prompt, answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vectorstore, app.state.embedding_model = load_index()
    app.state.prompt = build_prompt()
    yield


app = FastAPI(lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    results = retrieve(request.question, app.state.vectorstore, k=3)
    context = "\n---\n".join(r.page_content for r in results)
    message = answer_question(request.question, context, app.state.prompt)
    return AskResponse(answer=message.content[0].text)
