"""Wikipedia search/lookup environment for HotpotQA/FEVER, mirroring the
official ReAct implementation's WikiEnv (github.com/ysymyth/ReAct,
wikienv.py) as closely as possible: search[entity] scrapes the Wikipedia
search results page (matching disambiguation/"may refer to" handling and
mismatch fallback), lookup[keyword] cycles through sentences containing the
keyword in the current page. We add response caching on top (not present
upstream) to reduce repeated Wikipedia load across reruns."""

import json
import os

import requests
from bs4 import BeautifulSoup

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "wiki_cache.json")
_HEADERS = {"User-Agent": "ReAct-Reproduction-Research/1.0 (educational assignment; contact: local)"}


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _clean_str(p: str) -> str:
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def _get_page_obs(page: str) -> str:
    paragraphs = [p.strip() for p in page.split("\n") if p.strip()]
    sentences = []
    for p in paragraphs:
        sentences += p.split(". ")
    sentences = [s.strip() + "." for s in sentences if s.strip()]
    return " ".join(sentences[:5])


class WikiEnv:
    """Session state for one episode: current page text and the last lookup
    keyword/cursor, so repeated lookup[x] calls advance through successive
    matches, matching the official WikiEnv semantics."""

    def __init__(self):
        self._cache = _load_cache()
        self.page: str | None = None
        self.lookup_keyword: str | None = None
        self.lookup_list: list[str] | None = None
        self.lookup_cnt: int = 0

    def reset(self):
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = 0

    def _construct_lookup_list(self, keyword: str) -> list[str]:
        if self.page is None:
            return []
        paragraphs = [p.strip() for p in self.page.split("\n") if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return [s for s in sentences if keyword.lower() in s.lower()]

    def search(self, entity: str) -> str:
        cache_key = f"search::{entity.lower()}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            self.page = cached["page"]
            self.lookup_keyword = None
            self.lookup_list = None
            self.lookup_cnt = 0
            return cached["observation"]

        observation, page = self._search_live(entity)
        self._cache[cache_key] = {"observation": observation, "page": page}
        _save_cache(self._cache)
        self.page = page
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = 0
        return observation

    def _search_live(self, entity: str, _redirected: bool = False) -> tuple[str, str | None]:
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = requests.get(search_url, timeout=10, headers=_HEADERS).text
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

        if result_divs:
            titles = [_clean_str(div.get_text().strip()) for div in result_divs]
            observation = f"Could not find {entity}. Similar: {titles[:5]}."
            return observation, None

        content = soup.find("div", {"id": "mw-content-text"}) or soup
        page_bits = [p.get_text().strip() for p in content.find_all("p") + content.find_all("ul")]
        if any("may refer to:" in p for p in page_bits) and not _redirected:
            return self._search_live("[" + entity + "]", _redirected=True)

        page = ""
        for p in page_bits:
            if len(p.split(" ")) > 2:
                page += _clean_str(p)
                if not p.endswith("\n"):
                    page += "\n"
        observation = _get_page_obs(page)
        return observation, page

    def lookup(self, keyword: str) -> str:
        if keyword != self.lookup_keyword:
            self.lookup_keyword = keyword
            self.lookup_list = self._construct_lookup_list(keyword)
            self.lookup_cnt = 0
        if self.lookup_list is None or self.lookup_cnt >= len(self.lookup_list):
            return "No more results.\n"
        result = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) {self.lookup_list[self.lookup_cnt]}"
        self.lookup_cnt += 1
        return result

    def step(self, kind: str, arg: str) -> str:
        if kind == "search":
            return self.search(arg)
        if kind == "lookup":
            return self.lookup(arg)
        raise ValueError(f"WikiEnv cannot handle action kind: {kind}")
