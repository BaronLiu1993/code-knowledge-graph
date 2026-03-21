import base64
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")


def _headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def create_branch(owner, repository, branch_name, base_branch="main"):
    base_sha = _get_branch_sha(owner, repository, base_branch)
    response = requests.post(
        f"https://api.github.com/repos/{owner}/{repository}/git/refs",
        headers=_headers(),
        json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
    )
    response.raise_for_status()
    return response.json()


def create_or_update_file(owner, repository, path, content, message, branch="main"):
    url = f"https://api.github.com/repos/{owner}/{repository}/contents/{path}"
    existing_sha = _get_file_sha(owner, repository, path, branch)

    body = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if existing_sha:
        body["sha"] = existing_sha

    response = requests.put(url, headers=_headers(), json=body)
    response.raise_for_status()
    return response.json()


def create_pull_request(owner, repository, branch_name, title, body, base_branch="main"):
    response = requests.post(
        f"https://api.github.com/repos/{owner}/{repository}/pulls",
        headers=_headers(),
        json={
            "title": title,
            "body": body,
            "head": branch_name,
            "base": base_branch,
        },
    )
    response.raise_for_status()
    return response.json()


def _get_branch_sha(owner, repository, branch):
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repository}/git/ref/heads/{branch}",
        headers=_headers(),
    )
    response.raise_for_status()
    return response.json()["object"]["sha"]


def _get_file_sha(owner, repository, path, branch="main"):
    url = f"https://api.github.com/repos/{owner}/{repository}/contents/{path}"
    response = requests.get(url, headers=_headers(), params={"ref": branch})
    if response.status_code == 200:
        return response.json().get("sha")
    return None
