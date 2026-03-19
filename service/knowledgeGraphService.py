import json
import os
import requests
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Replace with OAuth later
token = os.getenv('GITHUB_ACCESS_TOKEN')
anthropic_client = anthropic.Client()

class GraphNode:
    def __init__(self, name, node_type):
        self.name = name
        self.node_type = node_type
        self.edges = []
    


def construct_relationship():
    pass

def get_repo_tree(owner, repository):
    HEADERS = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repository}/git/trees/main",
        headers=HEADERS,
        params={"recursive": "1"}
    )
    print(json.dumps(response.json(), indent=2))

def parse_repo_tree(raw_tree):
    
    pass

get_repo_tree("baronliu1993", "palettebackend")