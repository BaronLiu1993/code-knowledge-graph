from pydantic import BaseModel
from typing import List, Optional

class GithubRepositoryResponse(BaseModel):
    owner: str
    repository: str