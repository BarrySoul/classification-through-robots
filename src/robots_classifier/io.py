from __future__ import annotations

from pathlib import Path
import requests


def read_text_file(path: str | Path) -> str:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def fetch_robots_txt(url: str, timeout: int = 10) -> str:
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url

    robots_url = url.rstrip("/") + "/robots.txt"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/plain,text/*;q=0.9,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        "Referer": url,
    }

    r = requests.get(robots_url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.text