from pydantic import BaseModel

class GithubRepositoryResponse(BaseModel):
    owner: str
    repository: str