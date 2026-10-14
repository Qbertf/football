import os
import time
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.github.com"


def _resolve_token(token: str = None) -> str:
    """Return the provided token, or fall back to GITHUB_TOKEN env var."""
    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "No token provided. Pass token='ghp_...' or set the "
            "GITHUB_TOKEN environment variable."
        )
    return token


def make_session(token: str):
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    })
    return session


def _get_owner(session):
    r = session.get(f"{API}/user", timeout=30)
    r.raise_for_status()
    return r.json()["login"]


def create_repo_with_branches(
    repo_name: str,
    token: str = None,
    description: str = "",
    private: bool = True,
    num_branches: int = 20,
    branch_prefix: str = "branch_part_",
    start_index: int = 1,
):
    """
    Create a new repo and populate it with `num_branches` branches.

    Args:
        repo_name: Name of the new repository.
        token: GitHub PAT. Falls back to GITHUB_TOKEN env var if None.
        description: Optional description.
        private: Whether the repo should be private.
        num_branches: How many branches to create.
        branch_prefix: Prefix for branch names.
        start_index: Starting number for branch suffixes.

    Returns:
        dict with repo info, created branches, and failed branches.
    """
    token = _resolve_token(token)
    session = make_session(token)

    # 1) Authenticated user
    r = session.get(f"{API}/user", timeout=30)
    r.raise_for_status()
    owner = r.json()["login"]
    print(f"👤 Authenticated as: {owner}")

    # 2) Create repo
    r = session.post(
        f"{API}/user/repos",
        json={
            "name": repo_name,
            "description": description,
            "private": private,
            "auto_init": True,
        },
        timeout=30,
    )
    if r.status_code == 201:
        repo = r.json()
        print(f"✅ Repository '{repo['full_name']}' created.")
    elif r.status_code == 422:
        print("⚠️ Repository already exists. Fetching...")
        r = session.get(f"{API}/repos/{owner}/{repo_name}", timeout=30)
        r.raise_for_status()
        repo = r.json()
    else:
        r.raise_for_status()

    # 3) Wait for auto_init commit
    time.sleep(3)

    # 4) Base SHA
    default_branch = repo["default_branch"]
    r = session.get(
        f"{API}/repos/{owner}/{repo_name}/git/ref/heads/{default_branch}",
        timeout=30,
    )
    r.raise_for_status()
    base_sha = r.json()["object"]["sha"]
    print(f"🌱 Base branch '{default_branch}' at {base_sha[:7]}")

    # 5) Create branches
    created, failed = [], []
    for i in range(start_index, start_index + num_branches):
        branch_name = f"{branch_prefix}{i}"
        payload = {"ref": f"refs/heads/{branch_name}", "sha": base_sha}

        try:
            r = session.post(
                f"{API}/repos/{owner}/{repo_name}/git/refs",
                json=payload,
                timeout=30,
            )
            if r.status_code == 201:
                print(f"  ✅ Created: {branch_name}")
                created.append(branch_name)
            elif r.status_code == 422:
                print(f"  ⚠️ Already exists: {branch_name}")
                created.append(branch_name)
            else:
                print(f"  ❌ Failed {branch_name}: {r.status_code} {r.text}")
                failed.append(branch_name)
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Network error on {branch_name}: {e}")
            failed.append(branch_name)

        time.sleep(0.3)

    # 6) Retry failed ones once more
    if failed:
        print(f"\n🔁 Retrying {len(failed)} failed branches after 5s...")
        time.sleep(5)
        still_failed = []
        for branch_name in failed:
            payload = {"ref": f"refs/heads/{branch_name}", "sha": base_sha}
            try:
                r = session.post(
                    f"{API}/repos/{owner}/{repo_name}/git/refs",
                    json=payload,
                    timeout=30,
                )
                if r.status_code in (201, 422):
                    print(f"  ✅ Created on retry: {branch_name}")
                    created.append(branch_name)
                else:
                    still_failed.append(branch_name)
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Still failing: {branch_name} ({e})")
                still_failed.append(branch_name)
        failed = still_failed

    return {
        "repo": repo["full_name"],
        "url": repo["html_url"],
        "default_branch": default_branch,
        "created": created,
        "failed": failed,
    }

def repo_exists(
    repo_name: str,
    token: str = None,
    owner: str = None,
) -> dict:
    """
    Check whether a repository exists and is accessible.

    Args:
        repo_name: Repository name (e.g. "my-repo2").
        token: GitHub PAT. Falls back to GITHUB_TOKEN env var if None.
        owner: Repo owner. Defaults to the authenticated user.

    Returns:
        dict with:
            - exists (bool)
            - status (int)          # HTTP status code returned
            - full_name (str|None)
            - url (str|None)
            - default_branch (str|None)
            - private (bool|None)
            - message (str)         # human-readable explanation
    """
    token = _resolve_token(token)
    session = make_session(token)

    if owner is None:
        owner = _get_owner(session)

    r = session.get(f"{API}/repos/{owner}/{repo_name}", timeout=30)

    if r.status_code == 200:
        data = r.json()
        return {
            "exists": True,
            "status": 200,
            "full_name": data["full_name"],
            "url": data["html_url"],
            "default_branch": data["default_branch"],
            "private": data["private"],
            "message": f"Repository '{data['full_name']}' exists.",
        }

    if r.status_code == 404:
        return {
            "exists": False,
            "status": 404,
            "full_name": None,
            "url": None,
            "default_branch": None,
            "private": None,
            "message": f"Repository '{owner}/{repo_name}' does not exist "
                       f"(or token has no access).",
        }

    # Any other status: 401, 403, 5xx, etc.
    return {
        "exists": False,
        "status": r.status_code,
        "full_name": None,
        "url": None,
        "default_branch": None,
        "private": None,
        "message": f"Unexpected response {r.status_code}: {r.text[:200]}",
    }
    

def push(
    branch: str,
    list_files: list,
    repo_name: str,
    token: str = None,
    owner: str = None,
    commit_message: str = None,
    target_dir: str = "",
):
    """
    Push a list of local files to a specific branch by creating ONE commit.

    Args:
        branch: Branch name (e.g. "branch_part_1").
        list_files: List of local file paths to upload.
        repo_name: Repository name.
        token: GitHub PAT. Falls back to GITHUB_TOKEN env var if None.
        owner: Repo owner (defaults to authenticated user).
        commit_message: Commit message (defaults to auto-generated).
        target_dir: Optional subdirectory inside the repo (e.g. "data/").

    Returns:
        dict with commit SHA, branch, and uploaded paths.
    """
    token = _resolve_token(token)
    session = make_session(token)

    if owner is None:
        owner = _get_owner(session)

    if commit_message is None:
        commit_message = f"Add {len(list_files)} file(s) to {branch}"

    base = f"{API}/repos/{owner}/{repo_name}"

    # 1) Get the current commit SHA of the branch
    r = session.get(f"{base}/git/ref/heads/{branch}", timeout=30)
    if r.status_code == 404:
        raise ValueError(f"Branch '{branch}' not found in {owner}/{repo_name}.")
    r.raise_for_status()
    parent_sha = r.json()["object"]["sha"]

    # 2) Get the tree SHA of the parent commit
    r = session.get(f"{base}/git/commits/{parent_sha}", timeout=30)
    r.raise_for_status()
    base_tree_sha = r.json()["tree"]["sha"]

    # 3) Create blobs for each file
    tree_items = []
    uploaded_paths = []

    for local_path in list_files:
        if not os.path.exists(local_path):
            print(f"⚠️ Skipping missing file: {local_path}")
            continue

        filename = os.path.basename(local_path)
        repo_path = f"{target_dir.rstrip('/')}/{filename}" if target_dir else filename

        with open(local_path, "rb") as f:
            content_bytes = f.read()

        encoded = base64.b64encode(content_bytes).decode("utf-8")

        r = session.post(
            f"{base}/git/blobs",
            json={"content": encoded, "encoding": "base64"},
            timeout=60,
        )
        r.raise_for_status()
        blob_sha = r.json()["sha"]

        tree_items.append({
            "path": repo_path,
            "mode": "100644",
            "type": "blob",
            "sha": blob_sha,
        })
        uploaded_paths.append(repo_path)
        print(f"  📄 Blob created: {repo_path} ({len(content_bytes)} bytes)")

        time.sleep(0.2)

    if not tree_items:
        print("❌ No valid files to upload.")
        return None

    # 4) Create a new tree on top of the base tree
    r = session.post(
        f"{base}/git/trees",
        json={"base_tree": base_tree_sha, "tree": tree_items},
        timeout=60,
    )
    r.raise_for_status()
    new_tree_sha = r.json()["sha"]

    # 5) Create a commit
    r = session.post(
        f"{base}/git/commits",
        json={
            "message": commit_message,
            "tree": new_tree_sha,
            "parents": [parent_sha],
        },
        timeout=60,
    )
    r.raise_for_status()
    new_commit_sha = r.json()["sha"]

    # 6) Update the branch ref
    r = session.patch(
        f"{base}/git/refs/heads/{branch}",
        json={"sha": new_commit_sha, "force": False},
        timeout=30,
    )
    r.raise_for_status()

    print(f"🎉 Pushed {len(uploaded_paths)} file(s) to '{branch}' "
          f"(commit {new_commit_sha[:7]})")

    return {
        "owner": owner,
        "repo": repo_name,
        "branch": branch,
        "commit": new_commit_sha,
        "files": uploaded_paths,
    }