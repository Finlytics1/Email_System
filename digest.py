"""
Daily research digest: AI, Computational Finance and Privacy.
Fetches from arXiv, Hugging Face Daily Papers and a few RSS feeds, scores items against
keyword lists, and emails a short digest sized for a 10-30 minute morning read.

Run locally for a test:   DRY_RUN=1 python digest.py
"""
import html
import json
import os
import re
import smtplib
import ssl
from calendar import timegm
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import feedparser
import requests

# =============================== CONFIG ===============================
LOOKBACK_DAYS = 3            # 3 covers weekends; already-seen items are skipped anyway
READING_MINUTES = 25         # your reading budget (10-30)
MINUTES_PER_ITEM = 3         # rough time per item
ITEMS_PER_TRACK = 3          # max per track
SUMMARY_MAX_CHARS = 280      # keep summaries short
MIN_SCORE = 3                # ignore weak matches
FEED_BONUS = 2               # lets blog/news items compete with the many papers
STATE_DIR = Path("state")

# keywords, and a weight for how much you care about the track
TRACKS = {
    "Computational finance": {
        "weight": 1.0,
        "keywords": [
            "option pricing", "deep hedging", "monte carlo", "neural sde",
            "volatility", "risk model", "portfolio optimization", "derivatives",
            "algorithmic trading", "market simulation", "neural network option pricing",
            "black-scholes", "physics-informed",
        ],
    },
    "Privacy": {
        "weight": 1.0,
        "keywords": [
            "differential privacy", "federated learning", "gdpr", "synthetic data",
            "data leakage", "privacy-preserving", "data protection", "anonymization",
            "membership inference", "privacy attack", "model inversion",
        ],
    },
    "AI": {
        "weight": 1.0,
        "keywords": [
            "large language model", "foundation model", "agent",
            "reinforcement learning", "diffusion model", "reasoning", "alignment",
        ],
    },
}

ARXIV_KEYWORD_CATS = ["cs.LG", "cs.AI", "cs.CR", "stat.ML", "cs.CY"]
ARXIV_FINANCE_CATS = ["q-fin.CP", "q-fin.MF", "q-fin.ST", "q-fin.RM", "q-fin.PM"]

# Best-effort feed URLs. Dead feeds are skipped with a warning in the log; fix or remove them.
RSS_FEEDS = {
    "Quantocracy": "https://quantocracy.com/feed/",
    "EFF": "https://www.eff.org/rss/updates.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "OpenAI": "https://openai.com/news/rss.xml",
    "EDPB": "https://www.edpb.europa.eu/feed/news_en",
    "BIS": "https://www.bis.org/doclist/bis_fsi_publs.rss",
}
# =====================================================================

UA = {"User-Agent": "research-digest/1.0 (personal student project)"}
SEEN_FILE = STATE_DIR / "seen.json"
TREND_FILE = STATE_DIR / "trends.json"
DRY_RUN = os.getenv("DRY_RUN") == "1"


# ------------------------------ helpers ------------------------------
def load_json(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True))


def strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def shorten(text, limit):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sentences[:2]:
        if out and len(out) + len(s) > limit:
            break
        out = f"{out} {s}".strip()
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0] + "…"
    return out


def to_dt(struct):
    if not struct:
        return None
    return datetime.fromtimestamp(timegm(struct), tz=timezone.utc)


def norm_link(link):
    return re.sub(r"v\d+$", "", link.strip().rstrip("/"))


# ------------------------------ fetchers ------------------------------
def fetch_arxiv(cutoff):
    items = []
    queries = []

    fin = " OR ".join(f"cat:{c}" for c in ARXIV_FINANCE_CATS)
    queries.append(f"({fin})")

    cats = " OR ".join(f"cat:{c}" for c in ARXIV_KEYWORD_CATS)
    for track in TRACKS.values():
        kws = " OR ".join(f'all:"{k}"' for k in track["keywords"])
        queries.append(f"({cats}) AND ({kws})")

    for q in queries:
        try:
            r = requests.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": q,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": 40,
                },
                headers=UA,
                timeout=30,
            )
            r.raise_for_status()
            for e in feedparser.parse(r.text).entries:
                pub = to_dt(e.get("published_parsed"))
                if pub and pub >= cutoff:
                    items.append({
                        "title": strip_html(e.title),
                        "link": norm_link(e.link),
                        "summary": strip_html(e.get("summary", "")),
                        "published": pub,
                        "source": "arXiv",
                        "bonus": 0,
                    })
        except Exception as ex:
            print(f"[warn] arXiv query failed: {ex}")
    return items


def fetch_hf():
    items = []
    try:
        r = requests.get("https://huggingface.co/api/daily_papers", headers=UA, timeout=30)
        r.raise_for_status()
        for row in r.json():
            p = row.get("paper", {})
            if not p.get("id"):
                continue
            items.append({
                "title": strip_html(p.get("title", "")),
                "link": f"https://huggingface.co/papers/{p['id']}",
                "summary": strip_html(p.get("summary", "")),
                "published": datetime.now(timezone.utc),
                "source": "Hugging Face Daily",
                "bonus": min((p.get("upvotes") or 0) / 20, 3),
            })
    except Exception as ex:
        print(f"[warn] Hugging Face failed: {ex}")
    return items


def fetch_feeds(cutoff):
    items = []
    for name, url in RSS_FEEDS.items():
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            entries = feedparser.parse(r.content).entries
            if not entries:
                print(f"[warn] {name}: feed empty or unparseable")
            for e in entries:
                pub = to_dt(e.get("published_parsed") or e.get("updated_parsed"))
                if pub and pub < cutoff:
                    continue
                items.append({
                    "title": strip_html(e.get("title", "")),
                    "link": norm_link(e.get("link", "")),
                    "summary": strip_html(e.get("summary", "")),
                    "published": pub or datetime.now(timezone.utc),
                    "source": name,
                    "bonus": FEED_BONUS,
                })
        except Exception as ex:
            print(f"[warn] {name} feed failed: {ex}")
    return items


# ------------------------------ scoring ------------------------------
def score_item(item, now):
    title, body = item["title"].lower(), item["summary"].lower()
    best_track, best_score, hits = None, 0.0, set()
    for track, cfg in TRACKS.items():
        raw = 0
        for kw in cfg["keywords"]:
            in_t, in_b = kw in title, kw in body
            if in_t or in_b:
                hits.add(kw)
                raw += 3 if in_t else 1
        s = raw * cfg["weight"]
        if s > best_score:
            best_track, best_score = track, s
    if best_score == 0:
        return None, 0, hits
    recency = 1 if now - item["published"] < timedelta(days=1) else 0
    return best_track, best_score + item["bonus"] + recency, hits


def select(items, now):
    by_track = {t: [] for t in TRACKS}
    all_hits = {}
    for it in items:
        track, score, hits = score_item(it, now)
        if track and score >= MIN_SCORE:
            it["score"] = score
            by_track[track].append(it)
            for kw in hits:
                all_hits[kw] = all_hits.get(kw, 0) + 1
    picked = {}
    for t, lst in by_track.items():
        lst.sort(key=lambda x: x["score"], reverse=True)
        picked[t] = lst[:ITEMS_PER_TRACK]

    # trim to the reading budget, dropping the lowest scores first
    max_items = max(1, READING_MINUTES // MINUTES_PER_ITEM)
    flat = sorted(
        ((it["score"], t, it) for t, l in picked.items() for it in l), key=lambda x: x[0]
    )
    while len(flat) > max_items:
        _, t, it = flat.pop(0)
        picked[t].remove(it)
    return picked, all_hits


# ------------------------------ trends ------------------------------
def trend_line(trends, today):
    days = sorted(trends)
    if len(days) < 7:
        return f"Trend tracking is collecting a baseline (day {len(days)} of 7)."
    d = datetime.fromisoformat(today).date()
    recent, older = {}, {}
    for day, counts in trends.items():
        age = (d - datetime.fromisoformat(day).date()).days
        target = recent if age < 7 else older if age < 35 else None
        if target is None:
            continue
        for kw, c in counts.items():
            target[kw] = target.get(kw, 0) + c
    rising = []
    for kw, c in recent.items():
        base = older.get(kw, 0) / 4  # older window is 4 weeks
        if c >= 3:
            rising.append((c / max(base, 1), kw, c))
    rising.sort(reverse=True)
    if not rising:
        return "No clear trend shifts this week."
    parts = [f"{kw} ({c} mentions)" for _, kw, c in rising[:3]]
    return "Rising this week vs. the last month: " + ", ".join(parts) + "."


# ------------------------------ summaries ------------------------------
def llm_one_liners(picked):
    """Optional: one plain-English sentence per item if ANTHROPIC_API_KEY is set."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return {}
    flat = [it for lst in picked.values() for it in lst]
    if not flat:
        return {}
    payload = [
        {"id": str(i), "title": it["title"], "text": it["summary"][:800]}
        for i, it in enumerate(flat)
    ]
    prompt = (
        "For each item write ONE plain-English sentence (max 30 words) saying what it is and "
        "why it matters to an undergraduate interested in AI, computational finance and data "
        'privacy. Return only JSON like {"0": "sentence", "1": "sentence"}.\n\n'
        + json.dumps(payload)
    )
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        r.raise_for_status()
        text = r.json()["content"][0]["text"]
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        return {flat[int(k)]["link"]: v for k, v in data.items()}
    except Exception as ex:
        print(f"[warn] LLM summaries failed, using excerpts: {ex}")
        return {}


# ------------------------------ email ------------------------------
def render(picked, oneliners, trend, today):
    total = sum(len(v) for v in picked.values())
    minutes = total * MINUTES_PER_ITEM
    h = [
        '<div style="font-family:Arial,sans-serif;max-width:640px;margin:auto;color:#222">',
        f"<h2 style='margin-bottom:0'>Research digest, {today}</h2>",
        f"<p style='color:#666;margin-top:4px'>{total} items · about {minutes} min</p>",
    ]
    text = [f"Research digest, {today} ({total} items, about {minutes} min)\n"]
    for track, lst in picked.items():
        h.append(f"<h3 style='border-bottom:1px solid #ddd;padding-bottom:4px'>{track}</h3>")
        text.append(f"\n== {track} ==")
        if not lst:
            h.append("<p style='color:#888'>Nothing strong today.</p>")
            text.append("Nothing strong today.")
            continue
        for it in lst:
            summ = oneliners.get(it["link"]) or shorten(it["summary"], SUMMARY_MAX_CHARS)
            title = html.escape(it["title"])
            h.append(
                f"<p><a href='{html.escape(it['link'])}'><b>{title}</b></a><br>"
                f"<span style='color:#888;font-size:12px'>{it['source']} · "
                f"{it['published']:%d %b}</span><br>{html.escape(summ)}</p>"
            )
            text.append(f"- {it['title']} [{it['source']}]\n  {summ}\n  {it['link']}")
    h.append(f"<h3>Where things are heading</h3><p>{html.escape(trend)}</p></div>")
    text.append(f"\nTrend: {trend}")
    return "\n".join(h), "\n".join(text), minutes


def send_email(subject, html_body, text_body):
    sender = os.environ["GMAIL_ADDRESS"]
    password = os.environ["GMAIL_APP_PASSWORD"]
    to = os.environ["TO_EMAIL"]
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(sender, password)
        s.sendmail(sender, [to], msg.as_string())


# ------------------------------ main ------------------------------
def main():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    cutoff = now - timedelta(days=LOOKBACK_DAYS)

    items = fetch_arxiv(cutoff) + fetch_hf() + fetch_feeds(cutoff)
    print(f"Fetched {len(items)} raw items")

    seen = load_json(SEEN_FILE, {})
    unique, links = [], set()
    for it in items:
        if it["link"] and it["link"] not in seen and it["link"] not in links:
            links.add(it["link"])
            unique.append(it)
    print(f"{len(unique)} new after dedupe")

    picked, hits = select(unique, now)
    trends = load_json(TREND_FILE, {})
    trends[today] = hits
    trend = trend_line(trends, today)

    oneliners = llm_one_liners(picked)
    html_body, text_body, minutes = render(picked, oneliners, trend, today)
    subject = f"Research digest {today} (~{minutes} min)"

    if DRY_RUN:
        print(text_body)
        return

    send_email(subject, html_body, text_body)
    print("Email sent")

    # only record state after a successful send
    for lst in picked.values():
        for it in lst:
            seen[it["link"]] = today
    cutoff_day = (now - timedelta(days=60)).strftime("%Y-%m-%d")
    seen = {k: v for k, v in seen.items() if v >= cutoff_day}
    trends = {k: v for k, v in trends.items() if k >= cutoff_day}
    save_json(SEEN_FILE, seen)
    save_json(TREND_FILE, trends)


if __name__ == "__main__":
    main()
