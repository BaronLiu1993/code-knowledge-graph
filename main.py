from fastapi import FastAPI
from router.documentationAgentRouter import router as documentation_agent_router

app = FastAPI(title="Code Knowledge Graph API", version="1.0")
app.include_router(documentation_agent_router)