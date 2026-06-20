from fastapi import APIRouter, HTTPException
from models.agent import AgentRequest, AgentResponse
from services.agent import AgentServices
from routers.search import search_service

router = APIRouter()
agent_service = AgentServices(search_service=search_service)


@router.post("/agent", response_model=AgentResponse)
async def agent(request: AgentRequest):
    try:
        return await agent_service.analyze(request.query, request.limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
