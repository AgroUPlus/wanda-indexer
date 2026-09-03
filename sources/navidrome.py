"""Subsonic / Navidrome REST access."""
import hashlib
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Tuple

API_VERSION = "1.16.1"
CLIENT_NAME = "wanda-desktop-indexer"


def _auth_params(username: str, token_or_pass: str, salt: Optional[str] = None) -> dict:
    """Subsonic salted-token auth: t = md5(password + salt)."""
    if salt is None:
        salt = secrets.token_hex(6)
    token = hashlib.md5((token_or_pass + salt).encode("utf-8")).hexdigest()
    return {"u": username, "t": token, "s": salt, "v": API_VERSION, "c": CLIENT_NAME}


def build_subsonic_stream_url(base_url: str, username: str, token_or_pass: str,
                              song_id: str, salt: Optional[str] = None) -> str:
    """Builds an authenticated stream URL for one song."""
    params = _auth_params(username, token_or_pass, salt)
    # No `f=json` here: on success this endpoint returns audio, and asking for
    # JSON only changes how errors come back.
    params["id"] = song_id
    return f"{base_url.rstrip('/')}/rest/stream?{urllib.parse.urlencode(params)}"


def ping(base_url: str, username: str, token_or_pass: str,
         timeout: int = 20) -> Tuple[bool, str]:
    """Checks server reachability and credentials. Returns (ok, message).

    Called during preflight so bad credentials surface once, immediately, rather
    than as one opaque decode failure per track.
    """
    params = _auth_params(username, token_or_pass)
    params["f"] = "json"
    url = f"{base_url.rstrip('/')}/rest/ping?{urllib.parse.urlencode(params)}"

    try:
        request = urllib.request.Request(url, headers={"User-Agent": CLIENT_NAME})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {base_url}"
    except urllib.error.URLError as exc:
        return False, f"cannot reach {base_url}: {exc.reason}"
    except (OSError, ValueError) as exc:
        return False, f"unexpected response from {base_url}: {exc}"

    body = payload.get("subsonic-response", {})
    if body.get("status") == "ok":
        return True, f"{body.get('type', 'subsonic')} {body.get('serverVersion', '')}".strip()

    error = body.get("error", {})
    return False, error.get("message", "authentication failed")


def describe_error_body(raw: bytes) -> str:
    """If `raw` is a Subsonic JSON error rather than audio, describe it.

    Navidrome answers a bad request with HTTP 200 and a JSON body, so ffmpeg
    reports only "Invalid data found when processing input" -- useless on its
    own. Returns "" when the payload is not a Subsonic error.
    """
    if not raw or not raw.lstrip()[:1] == b"{":
        return ""
    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return ""
    body = payload.get("subsonic-response", {})
    error = body.get("error")
    if not error:
        return ""
    return f"{error.get('message', 'error')} (code {error.get('code', '?')})"
