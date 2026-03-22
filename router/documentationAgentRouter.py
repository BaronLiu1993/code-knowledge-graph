from fastapi import APIRouter

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])

# Process the current router and initialise the service
from service.documentationAgentService import generate_documentation, submit_documentation_pr
from service.knowledgeGraphService import build_knowledge_graph
from models.documentationAgentModel import GithubRepositoryResponse

@router.get("/build", response_model=GithubRepositoryResponse)
def build_graph(owner: str, repository: str):
    claude_files = generate_documentation(owner, repository)
    pr_url = submit_documentation_pr(owner, repository, claude_files)
    return {"status": "success", "pr_url": pr_url}