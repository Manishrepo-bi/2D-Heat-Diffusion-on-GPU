"""Team workflow client for the local GitHub emulator (Vercel Labs `emulate`).

The emulator gives us pull requests, reviews and merges at http://127.0.0.1:4000, but it
keeps everything in memory and does not check review rules. This tool adds:

  * disk-based saves : every action is appended to <records>/ledger.jsonl and can be
                       replayed after the emulator restarts (`restore`), plus snapshots (`save`)
  * team rules       : you cannot review your own pull request, and a merge needs an approval
                       from someone else with no open "changes requested"
  * a safe merge     : the owner's merge also merges the real git branch, runs the tests,
                       and pushes only if the tests pass

Who am I?   set GH_EMU_USER=participant1   (or participant2 / participant3)
Where?      set GH_EMU_URL=http://127.0.0.1:4000   (default)

Commands
    python tools/gh_emu.py check
    python tools/gh_emu.py pr-open    --head feature/numba-gpu --title "..." [--body "..."]
    python tools/gh_emu.py pr-list
    python tools/gh_emu.py pr-review  1 --approve | --request-changes  [--body "..."]
    python tools/gh_emu.py pr-comment 1 --body "..."
    python tools/gh_emu.py pr-merge   1              (participant1 only)
    python tools/gh_emu.py pr-merge   1 --no-review  (team without a reviewer: tests still gate the merge)
    python tools/gh_emu.py save
    python tools/gh_emu.py restore                    (after the emulator was restarted)
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OWNER = "participant1"
REPO = "2D-Heat-Diffusion-on-GPU"
BASE = f"/repos/{OWNER}/{REPO}"
SEED = Path(__file__).resolve().parent / "emulate.config.json"


def die(msg):
    print(f"STOP: {msg}")
    sys.exit(1)


def tokens_by_login():
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    return {v["login"]: tok for tok, v in seed["tokens"].items()}


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- HTTP
class Api:
    def __init__(self, url, login):
        toks = tokens_by_login()
        if login not in toks:
            die(f"GH_EMU_USER must be one of {sorted(toks)}, got {login!r}")
        self.url, self.login, self.token = url.rstrip("/"), login, toks[login]

    def as_user(self, login):
        return Api(self.url, login)

    def call(self, method, path, body=None, ok=(200, 201)):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Authorization": f"token {self.token}", "Accept": "application/vnd.github+json",
            "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                txt = r.read().decode()
                status = r.status
        except urllib.error.HTTPError as e:
            txt, status = e.read().decode(), e.code
        except urllib.error.URLError as e:
            die(f"cannot reach the GitHub emulator at {self.url} ({e.reason}). "
                f"Is tools\\start_emulator.bat running?")
        try:
            payload = json.loads(txt) if txt else None
        except ValueError:
            payload = {"message": txt}
        if ok and status not in ok:
            die(f"{method} {path} failed with HTTP {status}: {(payload or {}).get('message', payload)}")
        return status, payload


# ----------------------------------------------------------------------------- disk records
def git(*args, check=True):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        die(f"git {' '.join(args)} failed:\n{r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def records_dir(override=None):
    if override or os.environ.get("HEAT_RECORDS"):
        d = Path(override or os.environ["HEAT_RECORDS"])
    else:
        url = git("remote", "get-url", "origin", check=False)
        p = Path(url) if url else None
        if p is not None and p.is_dir():                 # disk-based remote: records sit beside it
            d = p.parent / "records"
        else:
            d = Path.cwd() / "emulator-records"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record(rdir, actor, action, **data):
    with open(rdir / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), "actor": actor, "action": action, "data": data}) + "\n")


# ----------------------------------------------------------------------------- helpers
def ensure_branch(api, branch):
    """Mirror a real git branch into the emulator so a pull request can point at it.

    The emulator only lets the repository owner create branches, and its branches carry no
    code (the code lives in our disk-based git remote). So the mirror is created with the
    owner's token, whoever runs the command."""
    owner = api.as_user(OWNER)
    status, _ = owner.call("GET", f"{BASE}/git/ref/heads/{branch}", ok=None)
    if status == 200:
        return False
    _, main = owner.call("GET", f"{BASE}/git/ref/heads/main")
    owner.call("POST", f"{BASE}/git/refs", {"ref": f"refs/heads/{branch}", "sha": main["object"]["sha"]})
    return True


def latest_reviews(api, number):
    _, reviews = api.call("GET", f"{BASE}/pulls/{number}/reviews")
    last = {}
    for r in reviews:                       # later reviews replace earlier ones per reviewer
        last[r["user"]["login"]] = r["state"]
    return last


# ----------------------------------------------------------------------------- commands
def cmd_check(api, a):
    _, user = api.call("GET", "/user")
    _, repo = api.call("GET", BASE)
    print(f"Connected to {api.url} as {user['login']}")
    print(f"Repository {repo['full_name']} found, default branch {repo['default_branch']}")
    print(f"Records folder: {records_dir(a.records)}")


def cmd_pr_open(api, a):
    if git("ls-remote", "--heads", "origin", a.head, check=False) == "":
        die(f"branch {a.head!r} is not on origin yet. Push it first: git push -u origin {a.head}")
    rdir = records_dir(a.records)
    if ensure_branch(api, a.head):
        record(rdir, api.login, "create_ref", branch=a.head)
    _, pr = api.call("POST", f"{BASE}/pulls", {"title": a.title, "head": a.head, "base": "main",
                                               "body": a.body or ""})
    record(rdir, api.login, "open_pr", number=pr["number"], title=a.title, head=a.head, body=a.body or "")
    print(f"Opened pull request #{pr['number']}: {a.title}  ({a.head} -> main) by {api.login}")


def cmd_pr_list(api, a):
    _, pulls = api.call("GET", f"{BASE}/pulls?state=all")
    if not pulls:
        print("No pull requests yet.")
    for p in sorted(pulls, key=lambda p: p["number"]):
        state = "merged" if p.get("merged") or p.get("merged_at") else p["state"]
        reviews = latest_reviews(api, p["number"])
        rv = ", ".join(f"{k}: {v}" for k, v in reviews.items()) or "no reviews"
        print(f"#{p['number']:<3} {state:<7} {p['head']['ref']:<24} by {p['user']['login']:<13} "
              f"{p['title']}   [{rv}]")


def cmd_pr_review(api, a):
    _, pr = api.call("GET", f"{BASE}/pulls/{a.number}")
    if pr["user"]["login"] == api.login:
        die("I cannot review my own pull request. Another participant has to do it.")
    if pr["state"] != "open":
        die(f"pull request #{a.number} is already {pr['state']}.")
    event = "APPROVE" if a.approve else "REQUEST_CHANGES"
    if event == "REQUEST_CHANGES" and not a.body:
        die("--request-changes needs --body explaining what to change.")
    api.call("POST", f"{BASE}/pulls/{a.number}/reviews", {"event": event, "body": a.body or ""})
    record(records_dir(a.records), api.login, "review", number=a.number, event=event, body=a.body or "")
    print(f"Review saved on #{a.number}: {event} by {api.login}")


def cmd_pr_comment(api, a):
    api.call("POST", f"{BASE}/issues/{a.number}/comments", {"body": a.body})
    record(records_dir(a.records), api.login, "comment", number=a.number, body=a.body)
    print(f"Comment added to #{a.number}")


def cmd_pr_merge(api, a):
    if api.login != OWNER:
        die(f"only {OWNER} merges pull requests in this project.")
    _, pr = api.call("GET", f"{BASE}/pulls/{a.number}")
    if pr["state"] != "open":
        die(f"pull request #{a.number} is already {pr['state']}.")
    author, head = pr["user"]["login"], pr["head"]["ref"]
    reviews = {k: v for k, v in latest_reviews(api, a.number).items() if k != author}
    if "CHANGES_REQUESTED" in reviews.values():
        die(f"changes were requested on #{a.number}: {reviews}. The author must fix them and the "
            f"reviewer must approve again.")
    if "APPROVED" not in reviews.values() and not a.no_review:
        die(f"#{a.number} has no approval from another participant yet: {reviews or 'no reviews'}. "
            f"(A team without a reviewer merges with --no-review.)")

    if not a.no_git:
        if git("status", "--porcelain"):
            die("my working folder has uncommitted changes. Commit or stash them first.")
        git("fetch", "origin")
        git("checkout", "main")
        git("pull", "--ff-only", "origin", "main")
        msg = f"Merge pull request #{a.number} from {head}: {pr['title']}"
        r = subprocess.run(["git", "merge", "--no-ff", f"origin/{head}", "-m", msg],
                           capture_output=True, text=True)
        if r.returncode != 0:
            subprocess.run(["git", "merge", "--abort"], capture_output=True)
            die(f"git could not merge {head} cleanly:\n{r.stdout}{r.stderr}")
        print("Merged locally. Running the tests before pushing...")
        t = subprocess.run([sys.executable, "-m", "pytest", "-q"], text=True)
        if t.returncode != 0:
            git("reset", "--hard", "ORIG_HEAD")
            die("tests failed after the merge, so I undid the merge. Nothing was pushed.")
        git("push", "origin", "main")
        print("Tests passed and main was pushed to origin.")
    api.call("PUT", f"{BASE}/pulls/{a.number}/merge", {"merge_method": "merge",
                                                         "commit_title": f"Merge #{a.number}"})
    record(records_dir(a.records), api.login, "merge", number=a.number, no_review=bool(a.no_review))
    print(f"Pull request #{a.number} marked as merged in the emulator.")


def cmd_save(api, a):
    rdir = records_dir(a.records)
    _, pulls = api.call("GET", f"{BASE}/pulls?state=all")
    snap = {"saved": now(), "url": api.url, "pulls": []}
    for p in pulls:
        _, reviews = api.call("GET", f"{BASE}/pulls/{p['number']}/reviews")
        _, comments = api.call("GET", f"{BASE}/issues/{p['number']}/comments")
        snap["pulls"].append({"number": p["number"], "title": p["title"], "head": p["head"]["ref"],
                              "author": p["user"]["login"], "state": p["state"],
                              "merged": bool(p.get("merged") or p.get("merged_at")),
                              "reviews": [(r["user"]["login"], r["state"], r.get("body", "")) for r in reviews],
                              "comments": [(c["user"]["login"], c["body"]) for c in comments]})
    path = rdir / f"snapshot-{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"Saved {len(snap['pulls'])} pull request(s) to {path}")


def cmd_restore(api, a):
    rdir = records_dir(a.records)
    ledger = rdir / "ledger.jsonl"
    if not ledger.exists():
        die(f"no ledger found at {ledger}")
    _, pulls = api.call("GET", f"{BASE}/pulls?state=all")
    if pulls:
        die("the emulator already has pull requests. Restore only right after a fresh start.")
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    for e in events:
        u, d = api.as_user(e["actor"]), e["data"]
        if e["action"] == "create_ref":
            ensure_branch(u, d["branch"])
        elif e["action"] == "open_pr":
            ensure_branch(u, d["head"])
            _, pr = u.call("POST", f"{BASE}/pulls", {"title": d["title"], "head": d["head"],
                                                     "base": "main", "body": d.get("body", "")})
            if pr["number"] != d["number"]:
                print(f"note: pull request was #{d['number']}, now #{pr['number']}")
        elif e["action"] == "review":
            u.call("POST", f"{BASE}/pulls/{d['number']}/reviews", {"event": d["event"], "body": d.get("body", "")})
        elif e["action"] == "comment":
            u.call("POST", f"{BASE}/issues/{d['number']}/comments", {"body": d["body"]})
        elif e["action"] == "merge":
            u.call("PUT", f"{BASE}/pulls/{d['number']}/merge", {"merge_method": "merge"})
    print(f"Replayed {len(events)} recorded action(s) from {ledger}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("GH_EMU_URL", "http://127.0.0.1:4000"))
    ap.add_argument("--user", default=os.environ.get("GH_EMU_USER"))
    ap.add_argument("--records", help="folder for disk-based records (default: next to the git remote)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("pr-open"); p.add_argument("--head", required=True)
    p.add_argument("--title", required=True); p.add_argument("--body")
    sub.add_parser("pr-list")
    p = sub.add_parser("pr-review"); p.add_argument("number", type=int)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true"); g.add_argument("--request-changes", action="store_true")
    p.add_argument("--body")
    p = sub.add_parser("pr-comment"); p.add_argument("number", type=int); p.add_argument("--body", required=True)
    p = sub.add_parser("pr-merge"); p.add_argument("number", type=int)
    p.add_argument("--no-git", action="store_true", help="only mark merged in the emulator")
    p.add_argument("--no-review", action="store_true", help="merge without an approval (tests must still pass)")
    sub.add_parser("save"); sub.add_parser("restore")
    a = ap.parse_args()
    if not a.user:
        die("set GH_EMU_USER first, for example:  set GH_EMU_USER=participant2")
    api = Api(a.url, a.user)
    {"check": cmd_check, "pr-open": cmd_pr_open, "pr-list": cmd_pr_list, "pr-review": cmd_pr_review,
     "pr-comment": cmd_pr_comment, "pr-merge": cmd_pr_merge, "save": cmd_save,
     "restore": cmd_restore}[a.cmd](api, a)


if __name__ == "__main__":
    main()
