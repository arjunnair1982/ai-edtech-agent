#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI EdTech Agent (single-file, dedupe+memory enhanced)

What it does:
- Fetches RSS items
- Extracts article text (robust HTTP retries)
- Filters for AI-in-education news (regex-safe, edtech-focused)
- Uses OpenAI to classify & summarize (with model cap & fallbacks)
- Dedupe (URL normalization + title + partial content hash)
- Optional ignore list (manual “never show again”)
- Email a digest (SendGrid SMTP-friendly)
- Optional fallback email if no items pass LLM

Env flags / settings (in .env):
  OPENAI_API_KEY=...
  OPENAI_MODEL=gpt-4o-mini
  MAX_LLM_CALLS=20
  DEBUG_FILTER=1
  DEBUG_DECISIONS=1
  SEND_TOP_CANDIDATES_IF_EMPTY=1
  FROM_NAME="AI EdTech Agent"
  SMTP_HOST=smtp.sendgrid.net
  SMTP_PORT=587
  SMTP_USER=apikey
  SMTP_PASS=... (SendGrid API key)
  FROM_EMAIL=you@domain
  TO_EMAIL=you@domain
"""

import os
import re
import json
import time
import hashlib
import logging
import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable, Optional, List

import feedparser
import requests
import trafilatura
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse, urlunparse, parse_qsl

# --------------------------
# Setup & constants
# --------------------------
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SEEN_PATH = DATA_DIR / "seen.json"
LOG_PATH = DATA_DIR / "run.log"
IGNORE_PATH = DATA_DIR / "ignore.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai-edtech-agent")

# Behavior flags
DEBUG_FILTER = os.getenv("DEBUG_FILTER", "0") == "1"
DEBUG_DECISIONS = os.getenv("DEBUG_DECISIONS", "0") == "1"
SEND_TOP_CANDIDATES_IF_EMPTY = os.getenv("SEND_TOP_CANDIDATES_IF_EMPTY", "0") == "1"
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "20"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_SOURCES = {
    "rss": [
        "https://techcrunch.com/feed/",
        "https://www.edsurge.com/feed",
        "https://thedecodr.com/feed"
    ],
    "websites": [
        "https://www.classcentral.com/report/ai/"
    ]
}

# --------------------------
# Regex-friendly term lists
# --------------------------
AI_TERMS: List[str] = [
    r"ai", r"artificial intelligence", r"genai", r"machine learning", r"ml"
]

EDU_TERMS: List[str] = [
    r"education(al)?", r"school(s)?", r"student(s)?", r"teacher(s)?",
    r"classroom(s)?", r"tutor(ing)?", r"curriculum", r"learning"
]

LAUNCH_FUNDING_TERMS: List[str] = [
    r"launch(ed|es|ing)?", r"unveil(ed|s|ing)?", r"introduc(e|ed|es|ing)",
    r"announce(d|s|ment|ments)?", r"raise(d|s)?", r"seed", r"series (a|b|c)",
    r"fund(ing)?", r"round"
]

SOFT_EDU: List[str] = [
    r"edtech", r"k-?12", r"tutor(ing)?", r"university", r"campus", r"mooc",
    r"lms", r"learning platform", r"school(s)?", r"student(s)?", r"teacher(s)?"
]

EXCLUDES: List[str] = [
    r"trump", r"election", r"congress", r"nasa", r"spacex", r"satellite",
    r"rocket", r"mars", r"bitcoin", r"crypto", r"ethereum", r"stock market"
]

def any_word(patterns: List[str], blob: str) -> bool:
    """True if ANY regex pattern matches as a whole word/phrase."""
    return any(re.search(rf"\b{p}\b", blob) for p in patterns)

# --------------------------
# HTTP session with retries
# --------------------------
def _http_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0 (ai-edtech-agent)"})
    return s

SESSION = _http_session()

def fetch_html(url: str, timeout: int = 12) -> str:
    """Fetch HTML with retries; return '' on failure."""
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.debug(f"HTML fetch failed for {url}: {e}")
        return ""

def extract_text(html: str, url: str) -> str:
    try:
        txt = trafilatura.extract(html, url=url)
        return txt or ""
    except Exception as e:
        logger.debug(f"Extraction failed for {url}: {e}")
        return ""

def trim_for_llm(txt: str, limit: int = 5000) -> str:
    return (txt or "")[:limit]

# --------------------------
# Dedupe helpers (NEW)
# --------------------------
def normalize_url(url: str) -> str:
    """Strip fragments and tracking params like utm_* so cosmetic URL changes don't bypass dedupe."""
    u = urlparse(url)
    clean_qs = [(k, v) for k, v in parse_qsl(u.query) if not k.lower().startswith("utm_")]
    return urlunparse((u.scheme, u.netloc, u.path, "", "&".join(f"{k}={v}" for k, v in clean_qs), ""))

def content_hash(text: str) -> str:
    """Hash of article content for stronger identity when we can extract it."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()

def item_key(url: str, title: str, text: str = "") -> str:
    """
    Strong identity:
    - normalized url
    - title (lowercased, trimmed)
    - first 16 chars of content hash (if available)
    """
    norm = normalize_url(url).lower()
    ttl = (title or "").strip().lower()
    ch = content_hash(text)[:16] if text else ""
    base = f"{norm}::{ttl}::{ch}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def load_seen() -> dict:
    """
    Returns dict of {key: iso_timestamp}. Handles old formats gracefully.
    """
    if SEEN_PATH.exists():
        try:
            data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                # migrate old list of keys -> dict with timestamps
                now = dt.datetime.now().isoformat()
                return {k: now for k in data}
        except Exception:
            pass
    return {}

def save_seen(seen: dict) -> None:
    SEEN_PATH.write_text(json.dumps(seen, indent=2), encoding="utf-8")

def load_ignore_list() -> list:
    if IGNORE_PATH.exists():
        return [ln.strip().lower() for ln in IGNORE_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return []

IGNORE_LIST = load_ignore_list()

# Optional: expire very old seen entries (disabled by default)
def purge_old_seen(seen: dict, days: int = 0) -> dict:
    if days <= 0:
        return seen
    cutoff = dt.datetime.now() - dt.timedelta(days=days)
    out = {}
    for k, ts in seen.items():
        try:
            if dt.datetime.fromisoformat(ts) >= cutoff:
                out[k] = ts
        except Exception:
            out[k] = dt.datetime.now().isoformat()
    return out

# --------------------------
# Data models
# --------------------------
class Finding(BaseModel):
    url: str
    title: str
    source: str
    company: Optional[str] = None
    stage: Optional[str] = None
    amount: Optional[str] = None
    product: Optional[str] = None
    summary: str
    why_it_matters: str

class LLMResp(BaseModel):
    is_relevant: bool
    company: Optional[str] = None
    product: Optional[str] = None
    stage: Optional[str] = None
    amount: Optional[str] = None
    one_sentence: str
    why_it_matters: str

# --------------------------
# Sources & dedupe (legacy helpers kept for compat)
# --------------------------
def fetch_rss_entries(feed_url: str, limit: int = 40) -> Iterable[dict]:
    d = feedparser.parse(feed_url)
    for e in d.entries[:limit]:
        yield {
            "title": e.get("title", "") or "",
            "link": e.get("link", "") or "",
            "summary": e.get("summary", "") or "",
            "published": e.get("published", "") or "",
            "source": feed_url
        }

def load_sources() -> dict:
    """Load sources.yaml if present; else defaults."""
    path = Path("sources.yaml")
    if path.exists():
        import yaml
        try:
            with path.open("r", encoding="utf-8") as f:
                return (yaml.safe_load(f) or DEFAULT_SOURCES)
        except Exception:
            logger.warning("Failed to parse sources.yaml, using defaults.")
    return DEFAULT_SOURCES

# --------------------------
# Prefilter (regex-safe, edtech-focused)
# --------------------------
def looks_like_candidate(title: str, text: str) -> bool:
    blob = f"{title}\n{text}".lower()

    if any_word(EXCLUDES, blob):
        if DEBUG_FILTER:
            logger.info(f"FILTER: excluded by EXCLUDES | title={title!r}")
        return False

    ai = any_word(AI_TERMS, blob)
    edu = any_word(EDU_TERMS, blob)
    lf  = any_word(LAUNCH_FUNDING_TERMS, blob)

    if not ai:
        if DEBUG_FILTER:
            logger.info(f"FILTER: no AI terms | title={title!r}")
        return False

    if edu:
        if DEBUG_FILTER:
            logger.info(f"FILTER: passed (AI+EDU) | title={title!r}")
        return True

    ok = lf and any_word(SOFT_EDU, blob)
    if DEBUG_FILTER:
        logger.info(
            f"FILTER: {'passed' if ok else 'blocked'} (AI+LF+SOFT_EDU={ok}) | title={title!r}"
        )
    return ok

def looks_like_launch_fund(title: str) -> bool:
    t = (title or "").lower()
    return any(re.search(rf"\b{p}\b", t) for p in AI_TERMS) and any(
        re.search(rf"\b{p}\b", t) for p in [r"launch", r"raise(d|s)?", r"seed", r"series (a|b|c)"]
    )

# --------------------------
# OpenAI integration
# --------------------------
_llm_calls = 0

def call_openai_json(prompt: str, max_retries: int = 2):
    """Call OpenAI; return parsed JSON dict or None on soft-fail (quota/rate)."""
    global _llm_calls
    if _llm_calls >= MAX_LLM_CALLS:
        logger.warning(f"LLM call cap reached ({MAX_LLM_CALLS}); skipping.")
        return None

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    last_err = None
    backoffs = [0.8, 2.0, 5.0]

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a precise analyst for AI-in-education product news. "
                                "Respond ONLY with valid JSON matching the schema."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text[text.find("{"): text.rfind("}") + 1]
            _llm_calls += 1
            return json.loads(text)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "insufficient_quota" in msg or "too many requests" in msg or "rate limit" in msg:
                logger.warning(f"OpenAI issue: {e}. Falling back to heuristic for this item.")
                return None
            if attempt < max_retries:
                time.sleep(backoffs[min(attempt, len(backoffs)-1)])
    logger.warning(f"OpenAI failed after retries: {last_err}")
    return None

def llm_extract_and_summarize(title: str, text: str) -> Optional[Finding]:
    """
    Returns Finding if relevant, else None.
    Falls back to a heuristic if JSON parsing/validation fails or call returns None.
    """
    prompt = f"""
Classify whether this article is about AI in education.
Set "is_relevant": true if ANY of these are true:
- It announces a NEW AI-in-education product/company.
- It reports a funding round for an AI-in-education company.
- It describes substantial AI-in-education product updates, pilots, or deployments (non-trivial).

Be strict about JSON. Do not include extra text.

Title: {title}

Article (trimmed):
{trim_for_llm(text)}

Return ONLY JSON with keys:
- is_relevant: boolean
- company: string or null
- product: string or null
- stage: string or null  (e.g., Seed, Series A)
- amount: string or null (e.g., "$5M")
- one_sentence: string (<=28 words)
- why_it_matters: string (<=28 words)
"""
    try:
        data = call_openai_json(prompt)
        if DEBUG_DECISIONS and data:
            logger.info(f"LLM decision: {data.get('is_relevant')} | one_sentence={data.get('one_sentence')!r}")
        if not data:
            raise json.JSONDecodeError("no data", "", 0)

        parsed = LLMResp(**data)
        if not parsed.is_relevant:
            return None
        return Finding(
            url="", title=title, source="",
            company=parsed.company, product=parsed.product,
            stage=parsed.stage, amount=parsed.amount,
            summary=parsed.one_sentence, why_it_matters=parsed.why_it_matters
        )
    except (ValidationError, json.JSONDecodeError):
        # Heuristic fallback so pipeline still runs
        if looks_like_launch_fund(title):
            return Finding(
                url="", title=title, source="",
                summary=title[:140],
                why_it_matters="Potentially relevant to AI-in-education; manual review suggested."
            )
        return None

# --------------------------
# Email output
# --------------------------
def format_email(findings: List[Finding]) -> str:
    lines = []
    today = dt.datetime.now().strftime("%Y-%m-%d")
    lines.append(f"AI-in-Education Daily — {today}\n")
    for i, f in enumerate(findings, 1):
        head = f"{i}. {f.title}"
        meta = []
        if f.company: meta.append(f.company)
        if f.stage: meta.append(f.stage)
        if f.amount: meta.append(f.amount)
        if meta:
            head += f" ({', '.join(meta)})"
        body = f"\n{f.summary}\nWhy it matters: {f.why_it_matters}\n{f.url}\n"
        lines.append(head + body)
    return "\n".join(lines)

def send_email(subject: str, body: str) -> None:
    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    TO_EMAIL = os.getenv("TO_EMAIL")
    FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER)
    FROM_NAME = os.getenv("FROM_NAME", "AI EdTech Agent")

    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, TO_EMAIL, FROM_EMAIL]):
        logger.warning("Email not sent: missing SMTP/.env settings.")
        return

    import smtplib
    from email.mime.text import MIMEText
    from email.utils import formataddr

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = formataddr((FROM_NAME, FROM_EMAIL))
    msg["To"] = TO_EMAIL
    # Helpful deliverability headers (optional):
    msg["Reply-To"] = FROM_EMAIL
    msg["List-Unsubscribe"] = f"<mailto:unsubscribe@{FROM_EMAIL.split('@')[-1]}>"

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

# --------------------------
# Main
# --------------------------
def run_agent(force_test: bool = False, max_per_feed: int = 40) -> None:
    # FAST PATH: force-test sends a dummy item and exits — NO OpenAI calls
    if force_test:
        logger.info("Force test enabled: sending dummy item without calling OpenAI.")
        found = [Finding(
            url="https://example.com",
            title="Test item: New AI tutor for K‑12 raises seed",
            source="test",
            company="Acme Tutors",
            stage="Seed",
            amount="$2.5M",
            product="AI tutoring platform",
            summary="Acme Tutors launches an AI tutoring platform for K‑12 and announces a seed round.",
            why_it_matters="Signals momentum in AI learning tools; potential for pilot discovery."
        )]
        email_body = format_email(found)
        send_email("AI-in-Education Daily", email_body)
        logger.info("Sent test email.")
        return

    seen = purge_old_seen(load_seen(), days=0)  # set days>0 to expire old entries
    found: List[Finding] = []
    candidates: List[Finding] = []
    sources = load_sources()
    rss_list = sources.get("rss", [])

    logger.info(f"Processing {len(rss_list)} RSS feeds...")

    for feed in rss_list:
        for e in fetch_rss_entries(feed, limit=max_per_feed):
            url = e["link"]
            title = e["title"] or "(no title)"
            if not url:
                continue

            # Manual ignore (title/url contains any phrase in ignore.txt)
            blob_for_ignore = f"{title}\n{url}".lower()
            if any(s in blob_for_ignore for s in IGNORE_LIST):
                pre_key = item_key(url, title, "")
                seen[pre_key] = dt.datetime.now().isoformat()
                if DEBUG_FILTER:
                    logger.info(f"FILTER: ignored by ignore.txt | title={title!r}")
                continue

            # EARLY KEY: URL+title (no content yet). Skip if we’ve already evaluated it.
            pre_key = item_key(url, title, "")
            if pre_key in seen:
                continue

            html = fetch_html(url)
            text = extract_text(html, url)

            # Require some content to avoid junk pages
            if len(text) < 400:
                if DEBUG_FILTER:
                    logger.info(f"FILTER: too-short text (<400 chars) | title={title!r}")
                # don't mark seen; might succeed next run if site was flaky
                continue

            # STRONG KEY: include content hash segment
            key = item_key(url, title, text)
            if key in seen:
                continue

            if not looks_like_candidate(title, text):
                # mark as seen so we don't reconsider reposts of non-relevant items
                seen[pre_key] = dt.datetime.now().isoformat()
                continue

            # Keep a lightweight candidate for potential fallback digest
            candidates.append(Finding(
                url=url,
                title=title,
                source=feed,
                summary=title[:140],
                why_it_matters="Keyword match suggests AI+education relevance; review."
            ))

            # LLM analysis (with fallback)
            item = llm_extract_and_summarize(title, text)

            if item:
                item.url = url
                item.source = feed
                found.append(item)

            # Mark seen using the strong key since we made a decision
            seen[key] = dt.datetime.now().isoformat()

    if found:
        email_body = format_email(found)
        send_email("AI-in-Education Daily", email_body)
        logger.info(f"Sent {len(found)} item(s).")
    else:
        logger.info("No relevant items today.")
        if SEND_TOP_CANDIDATES_IF_EMPTY and candidates:
            fallback = candidates[-3:]  # last 3 examined keyword matches
            logger.info(f"Sending fallback with {len(fallback)} candidate(s).")
            email_body = format_email(fallback)
            send_email("AI-in-Education Daily (fallback)", email_body)

    logger.info(f"Summary: feeds={len(rss_list)} candidates={len(candidates)} llm_calls={_llm_calls} findings={len(found)}")
    save_seen(seen)

def main():
    parser = argparse.ArgumentParser(description="AI EdTech Agent (dedupe+memory enhanced)")
    parser.add_argument("--force-test", action="store_true",
                        help="Send a dummy item to test email & pipeline (no OpenAI calls)")
    parser.add_argument("--max-per-feed", type=int, default=40,
                        help="Max entries per feed to scan")
    args = parser.parse_args()
    run_agent(force_test=args.force_test, max_per_feed=args.max_per_feed)

if __name__ == "__main__":
    main()