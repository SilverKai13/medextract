"""Fetch PubMed abstracts through NCBI E-utilities.

No key is needed at this volume, but NCBI publishes rate limits (3 requests a
second without a key, 10 with one) and asks for an identifying email. Both are
respected here -- getting a project's IP blocked from E-utilities is a slow
thing to undo.

    python -m medextract.retrieval.pubmed --query "lung neoplasms" --max 2000 \
        --out data/pubmed.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from medextract.config import REPO_ROOT, get_settings

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# efetch accepts more, but large id batches make failures expensive to retry.
BATCH_SIZE = 200


@dataclass
class Abstract:
    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    year: int | None = None

    @property
    def text(self) -> str:
        """Title plus abstract -- what gets embedded and what gets tagged."""
        return f"{self.title}\n\n{self.abstract}".strip()


class RateLimiter:
    """Minimum spacing between calls, applied by sleeping before each request."""

    def __init__(self, per_second: float) -> None:
        self.min_interval = 1.0 / per_second
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class PubMedClient:
    def __init__(
        self,
        email: str = "",
        api_key: str = "",
        *,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.email = email
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.limiter = RateLimiter(9.0 if api_key else 2.5)

    def _params(self, **extra) -> dict[str, str]:
        params = {"db": "pubmed", "tool": "medextract", **extra}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get(self, endpoint: str, params: dict[str, str], retries: int = 3) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(retries):
            self.limiter.wait()
            try:
                response = self.client.get(f"{EUTILS}/{endpoint}", params=params)
                # 429 is a rate-limit signal, not a permanent failure.
                if response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        "rate limited", request=response.request, response=response
                    )
                response.raise_for_status()
                return response
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                backoff = 2**attempt
                log.warning(
                    "%s failed (attempt %d), retrying in %ds", endpoint, attempt + 1, backoff
                )
                time.sleep(backoff)
        raise RuntimeError(f"{endpoint} failed after {retries} attempts") from last_error

    def search(self, query: str, max_results: int = 1000) -> list[str]:
        """esearch: query -> PMIDs."""
        pmids: list[str] = []
        retstart = 0
        while len(pmids) < max_results:
            retmax = min(BATCH_SIZE * 2, max_results - len(pmids))
            response = self._get(
                "esearch.fcgi",
                self._params(
                    term=query,
                    retmax=str(retmax),
                    retstart=str(retstart),
                    retmode="json",
                ),
            )
            batch = response.json().get("esearchresult", {}).get("idlist", [])
            if not batch:
                break
            pmids.extend(batch)
            retstart += len(batch)
        log.info("esearch %r -> %d pmids", query, len(pmids))
        return pmids[:max_results]

    def fetch(self, pmids: list[str]) -> list[Abstract]:
        """efetch: PMIDs -> parsed abstracts, skipping records with no abstract."""
        results: list[Abstract] = []
        for start in range(0, len(pmids), BATCH_SIZE):
            chunk = pmids[start : start + BATCH_SIZE]
            response = self._get(
                "efetch.fcgi",
                self._params(id=",".join(chunk), retmode="xml", rettype="abstract"),
            )
            results.extend(parse_pubmed_xml(response.text))
            log.info("fetched %d/%d", min(start + BATCH_SIZE, len(pmids)), len(pmids))
        return results

    def close(self) -> None:
        self.client.close()


def parse_pubmed_xml(xml_text: str) -> list[Abstract]:
    """Parse a PubmedArticleSet.

    Structured abstracts split the text across several AbstractText elements
    with Label attributes ("BACKGROUND", "METHODS"); those are joined back into
    one passage, keeping the labels, rather than only the first section being
    taken.
    """
    root = ET.fromstring(xml_text)
    abstracts: list[Abstract] = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        parts: list[str] = []
        for node in article.findall(".//Abstract/AbstractText"):
            body = "".join(node.itertext()).strip()
            if not body:
                continue
            label = node.get("Label")
            parts.append(f"{label}: {body}" if label else body)

        if not parts:
            # Records with no abstract cannot be embedded usefully.
            continue

        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//Journal/JournalIssue/PubDate/Year")
        year = None
        if year_el is not None and year_el.text and year_el.text.isdigit():
            year = int(year_el.text)

        abstracts.append(
            Abstract(
                pmid=pmid_el.text.strip(),
                title=title.strip(),
                abstract=" ".join(parts),
                journal=(
                    journal_el.text.strip() if journal_el is not None and journal_el.text else None
                ),
                year=year,
            )
        )

    return abstracts


def write_jsonl(abstracts: list[Abstract], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for abstract in abstracts:
            handle.write(json.dumps(asdict(abstract), ensure_ascii=False) + "\n")
    log.info("wrote %d abstracts to %s", len(abstracts), path)


def read_jsonl(path: Path) -> list[Abstract]:
    with path.open(encoding="utf-8") as handle:
        return [Abstract(**json.loads(line)) for line in handle if line.strip()]


DEFAULT_QUERY = (
    '(neoplasms[MeSH] OR "drug therapy"[MeSH] OR genetics[MeSH]) '
    "AND hasabstract AND english[lang] AND 2015:2025[dp]"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download PubMed abstracts via E-utilities.")
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--max", type=int, default=3000, dest="max_results")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "pubmed.jsonl")
    p.add_argument("--email", default=None, help="overrides MEDEXTRACT_NCBI_EMAIL")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ns = build_parser().parse_args(argv)
    settings = get_settings()

    client = PubMedClient(
        email=ns.email or settings.ncbi_email,
        api_key=settings.ncbi_api_key,
    )
    if not client.email:
        log.warning(
            "No contact email set. NCBI asks for one; export MEDEXTRACT_NCBI_EMAIL "
            "before running larger fetches."
        )
    try:
        pmids = client.search(ns.query, ns.max_results)
        abstracts = client.fetch(pmids)
    finally:
        client.close()

    write_jsonl(abstracts, ns.out)
    print(f"{len(abstracts)} abstracts -> {ns.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
