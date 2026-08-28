from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import agent, rag, search

app = FastAPI(title="Financial Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(rag.router)
app.include_router(agent.router)


@app.get("/")
def root():
    return {"status": "online"}
