from fastapi import APIRouter
from models.agent import AgentRequest, AgentResponse
from services.agent import AgentServices
from routers.search import search_service

router = APIRouter()
agent_service = AgentServices(search_service=search_service)


@router.post("/agent", response_model=AgentResponse)
async def agent(request: AgentRequest):
    return await agent_service.analyze(request.ticker, request.limit)
