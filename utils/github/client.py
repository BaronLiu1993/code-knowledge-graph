import base64
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".ico",
                   ".lock", ".md", ".txt", ".env", ".yaml", ".yml",
                   ".toml", ".cfg", ".ini", ".csv", ".log"}

def _headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def fetch_repo_tree(owner, repository):
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repository}/git/trees/main",
        headers=_headers(),
        params={"recursive": "1"}
    )
    response.raise_for_status()
    return response.json()

def fetch_file_content_by_sha(owner, repository, sha):
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repository}/git/blobs/{sha}",
        headers=_headers()
    )
    response.raise_for_status()
    encoded = response.json().get("content", "").replace("\n", "")
    return base64.b64decode(encoded).decode("utf-8", errors="replace")

def is_code_file(path):
    _, ext = os.path.splitext(path)
    return ext.lower() not in SKIP_EXTENSIONS and ext != ""
