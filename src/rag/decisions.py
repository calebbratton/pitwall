"""FIA stewards' decisions: per-car rulings (penalties, warnings, no further action) with the
stewards' reasons, scraped from the FIA documents site and indexed next to the regulations.

They answer "did / should X get a penalty?" with what the stewards actually decided, and give
precedent for an incident still under investigation ("impeding in practice: standard penalty is
a warning"). Only per-car rulings are kept ("Infringement - Car N - ..." / "Decision - Car N -
..."); bulk lap-time deletions, summonses, classifications and notes are skipped.

Usage: python scripts/ingest_decisions.py [--seasons 2026 2025]
PDFs are cached in data/decisions/pdf/, parsed records in data/decisions/records.json.
"""

import json
import re
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path

import httpx

BASE = "https://www.fia.com"
CHAMPIONSHIP = f"{BASE}/documents/championships/fia-formula-one-world-championship-14"
EVENT_LIST = f"{BASE}/decision-document-list/ajax/{{event_id}}"
CACHE = Path("data/decisions")
HEADERS = {"User-Agent": "Mozilla/5.0 (pitwall hobby project)"}

RULING = re.compile(r"^(Infringement|Decision) - Car (\d+)\b", re.IGNORECASE)
_HREF = re.compile(r'href="([^"]+\.pdf)"')
_DOC_TITLE = re.compile(r"Doc\s+(\d+)\s+-\s+([^<]+?)\s*<")
_PUBLISHED = re.compile(r'date-display-single">([^<]*)<')
_EVENT = re.compile(
    r'href="/decision-document-list/nojs/(?P<id>\d+)"[^>]*>\s*(?P<name>[^<]+?)\s*</a>'
)
_SEASON = re.compile(
    r'value="(?P<path>/documents/championships/[^"]*/season/season-(?P<y>\d{4})-\d+)"'
)
# Questions that stewards' rulings bear on (checked on the question and the router's queries).
STEWARDING = re.compile(
    r"penal|steward|investigat|imped|reprimand|warning|\bfined?\b|grid drop|disqualif|collision|"
    r"unsafe release|track limits|yellow flag|summon|infring|blocking|blocked",
    re.IGNORECASE,
)
FIELDS = (
    "No / Driver",
    "Competitor",
    "Time",
    "Session",
    "Fact",
    "Infringement",
    "Decision",
    "Reason",
)
MAX_REASON_CHARS = 1500


@dataclass(frozen=True)
class DocumentRef:
    season: int
    event: str  # FIA's event menu name (may differ from the venue: "Bahrain Grand Prix")
    number: int
    title: str
    url: str
    published: str  # "dd.mm.yy HH:MM" CET, as listed


@dataclass(frozen=True)
class StewardsDecision:
    season: int
    competition: str  # from the document header, e.g. "2026 Bahrain Grand Prix In Malaysia"
    number: int
    title: str
    url: str
    published: str
    driver: str
    competitor: str
    session: str
    fact: str
    infringement: str
    decision: str
    reason: str

    @property
    def key(self) -> str:
        """Citation key the analyst copies, e.g. "Doc 26, 2026 Bahrain Grand Prix In Malaysia"."""
        return f"Doc {self.number}, {self.competition}"

    @property
    def citation(self) -> str:
        return f"Stewards' decision: {self.competition}, Doc {self.number} - {self.title}"

    def text(self) -> str:
        reason = self.reason[:MAX_REASON_CHARS] + (
            "..." if len(self.reason) > MAX_REASON_CHARS else ""
        )
        return (
            f"{self.competition}, {self.session}. Driver: {self.driver} ({self.competitor}). "
            f"Fact: {self.fact} Infringement: {self.infringement} Decision: {self.decision} "
            f"Reason: {reason}"
        )


# --- listing ---------------------------------------------------------------------------


def parse_rows(html: str, season: int, event: str) -> list[DocumentRef]:
    """Document rows: the markup differs between events (plain divs vs Drupal field wrappers),
    so each row is read by its link, its "Doc N - ..." title and its published date."""
    out = []
    for row in html.split('<li class="document-row')[1:]:
        href, title = _HREF.search(row), _DOC_TITLE.search(row)
        if not (href and title):
            continue
        published = _PUBLISHED.search(row)
        url = href[1]
        out.append(
            DocumentRef(
                season=season,
                event=event,
                number=int(title[1]),
                title=" ".join(unescape(title[2]).split()),
                url=BASE + url if url.startswith("/") else url,
                published=published[1].strip() if published else "",
            )
        )
    return out


def parse_season_page(html: str, season: int) -> tuple[list[DocumentRef], list[tuple[str, str]]]:
    """(documents of the event the page opens on, [(event_id, name)] of the other events)."""
    active = re.search(r'event-title active">\s*([^<]+?)\s*<', html)
    docs = parse_rows(html, season, active[1] if active else "")
    events = [(m["id"], unescape(m["name"])) for m in _EVENT.finditer(html)]
    return docs, events


def ajax_html(payload: str) -> str:
    """The AJAX event list is a JSON array of Drupal commands; the rows are in their `data`."""
    return "".join(c.get("data") or "" for c in json.loads(payload) if isinstance(c, dict))


def _request(client: httpx.Client, method: str, url: str, **kw) -> httpx.Response:
    """The FIA site 502s / 503s now and then: retry a few times with backoff."""
    import time

    for attempt in range(4):
        r = client.request(method, url, **kw)
        if r.status_code < 500 or attempt == 3:
            r.raise_for_status()
            return r
        time.sleep(2 * 2**attempt)
    raise AssertionError("unreachable")


def season_pages(client: httpx.Client) -> dict[int, str]:
    html = _request(client, "GET", CHAMPIONSHIP).text
    return {int(m["y"]): BASE + m["path"] for m in _SEASON.finditer(html)}


def list_rulings(client: httpx.Client, season: int) -> list[DocumentRef]:
    url = season_pages(client).get(season)
    if url is None:
        raise LookupError(f"no FIA documents page for season {season}")
    docs, events = parse_season_page(_request(client, "GET", url).text, season)
    for event_id, name in events:
        r = _request(
            client,
            "POST",
            EVENT_LIST.format(event_id=event_id),
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        docs += parse_rows(ajax_html(r.text), season, name)
    return [d for d in docs if RULING.match(d.title)]


# --- parsing ---------------------------------------------------------------------------


def parse_decision(text: str, ref: DocumentRef) -> StewardsDecision:
    """Fields from a decision PDF's text: label lines ("Fact ...", "Decision ...") start a field
    that runs until the next label; the reason ends at the appeal boilerplate."""
    text = text.replace("\xa0", " ")  # the PDFs use non-breaking spaces ("No /\xa0Driver")
    lines = [ln.strip() for ln in text.splitlines()]
    competition = next((ln for ln in lines if ln), "").title()
    values: dict[str, list[str]] = {}
    current = None
    for ln in lines:
        if ln.startswith(("Competitors are reminded", "Decisions of the Stewards are taken")):
            current = None
            continue
        if ln == "The Stewards":  # page-break footer
            continue
        label = next((f for f in FIELDS if ln == f or ln.startswith(f + " ")), None)
        if label and label not in values:
            current = label
            values[label] = [ln[len(label) :].strip()]
        elif current:
            values[current].append(ln)

    def get(name: str) -> str:
        return " ".join(" ".join(values.get(name, [])).split())

    return StewardsDecision(
        season=ref.season,
        competition=competition,
        number=ref.number,
        title=ref.title,
        url=ref.url,
        published=ref.published,
        driver=re.sub(r"^\d+\s*-\s*", "", get("No / Driver")),
        competitor=get("Competitor"),
        session=get("Session"),
        fact=get("Fact"),
        infringement=get("Infringement"),
        decision=get("Decision"),
        reason=get("Reason"),
    )


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


# --- store -----------------------------------------------------------------------------


def load_records(cache: Path = CACHE) -> list[StewardsDecision]:
    path = cache / "records.json"
    if not path.exists():
        return []
    return [StewardsDecision(**r) for r in json.loads(path.read_text())]


def ingest(seasons: list[int], cache: Path = CACHE) -> list[StewardsDecision]:
    """Fetch rulings not seen before (by URL) for `seasons`; returns all records."""
    records = {r.url: r for r in load_records(cache)}
    pdfs = cache / "pdf"
    pdfs.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for season in seasons:
            for ref in list_rulings(client, season):
                if ref.url in records:
                    continue
                path = pdfs / ref.url.rsplit("/", 1)[-1]
                if not path.exists():
                    path.write_bytes(_request(client, "GET", ref.url).content)
                records[ref.url] = parse_decision(pdf_text(path), ref)
            _save(records, cache)  # per season, so a failure later keeps what's done
    return _save(records, cache)


def _save(records: dict[str, StewardsDecision], cache: Path) -> list[StewardsDecision]:
    out = sorted(records.values(), key=lambda r: (r.season, r.competition, r.number))
    (cache / "records.json").write_text(json.dumps([asdict(r) for r in out], indent=1))
    return out


# --- search ----------------------------------------------------------------------------

RRF_K = 60


class DecisionIndex:
    """Hybrid search (dense bge-small + BM25, reciprocal rank fusion) over the parsed rulings,
    in memory: a few hundred records embed in seconds, and keeping them out of the embedded
    Qdrant store lets the running API refresh them (Qdrant's local mode allows one process).
    Reloads by itself when records.json changes."""

    def __init__(self, cache: Path = CACHE, dense=None, sparse=None) -> None:
        self._path = cache / "records.json"
        self._dense, self._sparse = dense, sparse
        self._mtime = None
        self.records: list[StewardsDecision] = []

    def _models(self):
        if self._dense is None:
            from fastembed import SparseTextEmbedding, TextEmbedding

            from src.rag.index import DENSE_MODEL, SPARSE_MODEL

            self._dense, self._sparse = (
                TextEmbedding(DENSE_MODEL),
                SparseTextEmbedding(SPARSE_MODEL),
            )
        return self._dense, self._sparse

    def _refresh(self) -> None:
        import numpy as np

        mtime = self._path.stat().st_mtime if self._path.exists() else None
        if mtime == self._mtime:
            return
        self._mtime = mtime
        self.records = load_records(self._path.parent)
        if not self.records:
            return
        dense, sparse = self._models()
        texts = [f"{r.title}. {r.text()}" for r in self.records]
        vecs = np.array(list(dense.embed(texts)))
        self._vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self._docs = [
            dict(zip(s.indices.tolist(), s.values.tolist(), strict=True))
            for s in sparse.embed(texts)
        ]
        n = len(self._docs)
        df: dict[int, int] = {}
        for d in self._docs:
            for t in d:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: float(np.log(1 + (n - c + 0.5) / (c + 0.5))) for t, c in df.items()}

    def search(self, query: str, seasons: list[int], k: int = 3) -> list[StewardsDecision]:
        import numpy as np

        from src.rag.index import _QUERY_PREFIX

        self._refresh()
        allowed = [i for i, r in enumerate(self.records) if r.season in seasons]
        if not allowed:
            return []
        dense, sparse = self._models()
        q = next(dense.embed([_QUERY_PREFIX + query]))
        q = q / np.linalg.norm(q)
        cos = {i: float(self._vecs[i] @ q) for i in allowed}
        qs = next(sparse.query_embed(query))
        terms = qs.indices.tolist()
        bm25 = {
            i: sum(self._docs[i].get(t, 0.0) * self._idf.get(t, 0.0) for t in terms)
            for i in allowed
        }
        score: dict[int, float] = {}
        for ranking in (cos, {i: s for i, s in bm25.items() if s > 0}):
            for rank, i in enumerate(
                sorted(ranking, key=ranking.__getitem__, reverse=True)[: k * 4]
            ):
                score[i] = score.get(i, 0.0) + 1 / (RRF_K + rank)
        return [self.records[i] for i in sorted(score, key=score.__getitem__, reverse=True)[:k]]
