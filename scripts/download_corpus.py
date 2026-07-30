"""Download a license-clear corpus of recent arXiv cs.CL papers via OAI-PMH.

Why OAI-PMH and not the Search API: the Search API (`export.arxiv.org/api/query`) returns
HTTP 429 "Rate exceeded" on a cold first request from some networks, including the one this
was built on. OAI-PMH is the interface arXiv designates for bulk metadata harvesting, it
answers normally, and it has real flow control (a 503 with `Retry-After`) that a client can
cooperate with instead of guessing.

Other decisions that matter:

* The delay between requests is never reduced to "speed things up". arXiv blocks aggressive
  clients by IP, and a blocked IP costs far more time than the delay saves.
* A 503 with `Retry-After` is arXiv asking for a pause, not an error. It is honoured exactly.
* Full-text HTML exists for most recent papers but not all. When it is missing or the host
  throttles the request, the paper still lands abstract-only, flagged `full_text: false` in
  the manifest, and the summary reports the split. Papers are never silently dropped, and the
  PDF is never fetched as a fallback.
* The run is idempotent and resumable. A paper whose file already exists with a matching
  digest is skipped, and the manifest is rewritten after every document, so an interrupted
  run loses at most one paper.
* The texts are not committed to git, because arXiv papers carry per-paper licenses. The
  manifest with a SHA256 per file is committed, so a fresh clone can verify what it fetched.

Usage:
    uv run python scripts/download_corpus.py --limit 300
    uv run python scripts/download_corpus.py --limit 300 --with-html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import yaml

OAI_BASE = "https://export.arxiv.org/oai2"
HTML_BASE = "https://arxiv.org/html"

OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV_NS = "{http://arxiv.org/OAI/arXiv/}"

#: arXiv asks programmatic clients to space requests. Do not lower this.
DEFAULT_DELAY_SECONDS = 3.0
MAX_RETRIES = 5
#: Cap on a Retry-After we will actually wait out before giving up on the run.
MAX_RETRY_AFTER_SECONDS = 300
USER_AGENT = (
    "retrieval-engine-corpus-builder/0.1 "
    "(+https://github.com/akriti-adarsh/retrieval-engine; OAI-PMH harvester)"
)


@dataclass
class Paper:
    """One arXiv record as OAI-PMH reports it."""

    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    updated: str
    abstract: str
    categories: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class RateLimiter:
    """Global minimum spacing between outbound requests."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay = delay_seconds
        self._last: float | None = None

    @property
    def delay(self) -> float:
        return self._delay

    def wait(self) -> None:
        if self._last is not None:
            elapsed = time.monotonic() - self._last
            if elapsed < self._delay:
                time.sleep(self._delay - elapsed)
        self._last = time.monotonic()


def _retry_after(response: httpx.Response, fallback: float) -> float:
    """How long arXiv asked us to wait, clamped to something sane."""
    raw = response.headers.get("retry-after", "")
    try:
        seconds = float(raw)
    except ValueError:
        seconds = fallback
    return min(max(seconds, fallback), float(MAX_RETRY_AFTER_SECONDS))


def get_with_backoff(
    client: httpx.Client,
    url: str,
    limiter: RateLimiter,
    *,
    params: dict[str, str] | None = None,
    allow_404: bool = False,
) -> httpx.Response | None:
    """GET with flow control and exponential backoff. Returns None for an allowed 404.

    A 404 is a fact about the resource, not a transient failure, so it is never retried.
    """
    last_error = "no attempt made"
    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        try:
            response = client.get(url, params=params)
        except httpx.RequestError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 404 and allow_404:
                return None
            if response.status_code < 400:
                return response
            if response.status_code == 503:
                # arXiv's documented flow control. Cooperate rather than hammer.
                pause = _retry_after(response, limiter.delay)
                print(f"    arXiv asked for a {pause:.0f}s pause (503 flow control)")
                time.sleep(pause)
                last_error = "HTTP 503 flow control"
                continue
            last_error = f"HTTP {response.status_code}"
            if response.status_code != 429:
                break
        if attempt < MAX_RETRIES:
            sleep_for = limiter.delay * (2 ** (attempt - 1))
            print(f"    retry {attempt}/{MAX_RETRIES - 1} after {last_error}, {sleep_for:.0f}s")
            time.sleep(sleep_for)
    print(f"    giving up on {url}: {last_error}")
    return None


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def parse_records(xml_bytes: bytes) -> tuple[list[Paper], str | None]:
    """Parse one OAI-PMH ListRecords page into papers plus the resumption token."""
    # arXiv is a trusted, non-hostile source, so the stdlib parser is appropriate.
    root = ET.fromstring(xml_bytes)

    error = root.find(f"{OAI_NS}error")
    if error is not None:
        code = error.attrib.get("code", "unknown")
        print(f"    OAI error [{code}]: {_text(error)}")
        return [], None

    papers: list[Paper] = []
    for record in root.iter(f"{OAI_NS}record"):
        meta = record.find(f"{OAI_NS}metadata/{ARXIV_NS}arXiv")
        if meta is None:
            continue
        arxiv_id = _text(meta.find(f"{ARXIV_NS}id"))
        if not arxiv_id:
            continue
        authors: list[str] = []
        for author in meta.iter(f"{ARXIV_NS}author"):
            keyname = _text(author.find(f"{ARXIV_NS}keyname"))
            forenames = _text(author.find(f"{ARXIV_NS}forenames"))
            full = " ".join(part for part in (forenames, keyname) if part)
            if full:
                authors.append(full)
        categories = _text(meta.find(f"{ARXIV_NS}categories")).split()
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_text(meta.find(f"{ARXIV_NS}title")),
                authors=authors,
                published=_text(meta.find(f"{ARXIV_NS}created")),
                updated=_text(meta.find(f"{ARXIV_NS}updated")),
                abstract=_text(meta.find(f"{ARXIV_NS}abstract")),
                categories=categories,
            )
        )

    token_node = root.find(f"{OAI_NS}ListRecords/{OAI_NS}resumptionToken")
    token = _text(token_node) or None
    return papers, token


def harvest(
    client: httpx.Client,
    limiter: RateLimiter,
    category: str,
    limit: int,
    days_back: int,
) -> list[Paper]:
    """Harvest records from the ``cs`` set and keep those in ``category``.

    OAI-PMH sets are archive-level (``cs``), not subject-level, so the category filter is
    applied client-side against each record's own category list. There is also no sort
    option, so recency comes from the ``from`` date window rather than from a sort parameter.
    """
    since = (datetime.now(UTC) - timedelta(days=days_back)).date().isoformat()
    params: dict[str, str] = {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "set": "cs",
        "from": since,
    }
    kept: list[Paper] = []
    seen: set[str] = set()
    pages = 0

    while len(kept) < limit:
        pages += 1
        print(f"  page {pages} ({len(kept)}/{limit} kept, from {since})")
        response = get_with_backoff(client, OAI_BASE, limiter, params=params)
        if response is None:
            break
        papers, token = parse_records(response.content)
        if not papers and token is None:
            break
        for paper in papers:
            if paper.arxiv_id in seen:
                continue
            seen.add(paper.arxiv_id)
            if category in paper.categories and paper.abstract:
                kept.append(paper)
                if len(kept) >= limit:
                    break
        if token is None:
            print("  no resumption token, the window is exhausted")
            break
        # A resumption token replaces every other argument, per the OAI-PMH spec.
        params = {"verb": "ListRecords", "resumptionToken": token}

    return kept[:limit]


def html_to_markdown(html: str) -> str:
    """Flatten arXiv's HTML full text into markdown headings and paragraphs.

    Deliberately lossy: math, figures, and tables are dropped. What retrieval needs here is
    dense technical prose, and LaTeX artifacts would only add noise to the chunks.
    """
    from selectolax.lexbor import LexborHTMLParser

    tree = LexborHTMLParser(html)
    for selector in ("script", "style", "math", "figure", "table"):
        for node in tree.css(selector):
            node.decompose()

    parts: list[str] = []
    for node in tree.css("h1, h2, h3, h4, p, li"):
        text = " ".join((node.text() or "").split())
        if not text:
            continue
        tag = node.tag
        if tag in {"h1", "h2", "h3", "h4"}:
            parts.append(f"\n{'#' * (int(tag[1]) + 1)} {text}\n")
        elif tag == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def render_document(paper: Paper, body: str, *, full_text: bool) -> str:
    """Build the markdown file: YAML front matter, then abstract, then any full text."""
    front_matter = yaml.safe_dump(
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "published": paper.published,
            "url": paper.url,
            "categories": paper.categories,
            "full_text": full_text,
        },
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    sections = [f"---\n{front_matter}---", f"# {paper.title}", "## Abstract", paper.abstract]
    if body:
        sections.append(body)
    return "\n\n".join(sections).rstrip() + "\n"


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"documents": {}}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"  manifest at {path} is unreadable, starting fresh")
        return {"documents": {}}
    if not isinstance(loaded, dict) or not isinstance(loaded.get("documents"), dict):
        return {"documents": {}}
    return loaded


def write_manifest(path: Path, manifest: dict[str, object], category: str) -> None:
    manifest["generated_at"] = datetime.now(UTC).isoformat()
    manifest["category"] = category
    manifest["source"] = "arXiv OAI-PMH (export.arxiv.org/oai2)"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300, help="How many papers to keep.")
    parser.add_argument("--category", default="cs.CL", help="arXiv category to filter for.")
    parser.add_argument("--out", type=Path, default=Path("data/corpus"), help="Output dir.")
    parser.add_argument(
        "--days-back", type=int, default=45, help="How far back the harvest window reaches."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Minimum seconds between requests. Do not lower this; arXiv blocks by IP.",
    )
    parser.add_argument(
        "--with-html",
        action="store_true",
        help="Also try the full-text HTML. Slower, and not available for every paper.",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    documents = manifest["documents"]
    if not isinstance(documents, dict):
        documents = {}
        manifest["documents"] = documents

    limiter = RateLimiter(args.delay)
    headers = {"User-Agent": USER_AGENT}
    skipped = with_full_text = abstract_only = failed = 0

    with httpx.Client(timeout=120.0, headers=headers, follow_redirects=True) as client:
        print(f"Harvesting {args.limit} {args.category} papers from the last {args.days_back} days")
        papers = harvest(client, limiter, args.category, args.limit, args.days_back)
        if not papers:
            print("No papers harvested; nothing to do.")
            return 1
        print(f"Harvest returned {len(papers)} papers")

        for index, paper in enumerate(papers, start=1):
            target = out_dir / f"{paper.arxiv_id.replace('/', '_')}.md"
            existing = documents.get(paper.arxiv_id)
            if (
                target.exists()
                and isinstance(existing, dict)
                and existing.get("sha256") == sha256_of(target.read_text(encoding="utf-8"))
            ):
                skipped += 1
                if existing.get("full_text"):
                    with_full_text += 1
                else:
                    abstract_only += 1
                continue

            body = ""
            full_text = False
            if args.with_html:
                response = get_with_backoff(
                    client, f"{HTML_BASE}/{paper.arxiv_id}", limiter, allow_404=True
                )
                if response is not None:
                    body = html_to_markdown(response.text)
                    full_text = bool(body)

            if not paper.abstract and not body:
                failed += 1
                print(f"[{index}/{len(papers)}] {paper.arxiv_id}: no abstract and no body")
                continue

            rendered = render_document(paper, body, full_text=full_text)
            target.write_text(rendered, encoding="utf-8")
            documents[paper.arxiv_id] = {
                "file": target.name,
                "sha256": sha256_of(rendered),
                "full_text": full_text,
                "title": paper.title,
                "url": paper.url,
                "published": paper.published,
                "chars": len(rendered),
            }
            write_manifest(manifest_path, manifest, args.category)

            if full_text:
                with_full_text += 1
            else:
                abstract_only += 1
            kind = "full text" if full_text else "abstract"
            print(
                f"[{index}/{len(papers)}] {paper.arxiv_id} {kind}, "
                f"{len(rendered)} chars: {paper.title[:58]}"
            )

    write_manifest(manifest_path, manifest, args.category)
    print(
        f"\nDone. {len(documents)} documents in {out_dir} "
        f"({with_full_text} with full text, {abstract_only} abstract only, "
        f"{skipped} already present, {failed} failed). Manifest: {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
