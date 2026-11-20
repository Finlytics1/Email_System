"""
Daily research digest: ML in finance (fraud detection, insurance pricing, stock/asset
pricing, credit risk). About 6 papers, sized for a 20-minute read.
Fetches from arXiv, Hugging Face Daily Papers and a few RSS feeds, scores items against
keyword lists, and emails a short, nicely formatted digest.

Run locally for a test:   DRY_RUN=1 python digest.py
"""
import html
import json
import os
import re
import smtplib
import ssl
import time
from calendar import timegm
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import feedparser
import requests

# =============================== CONFIG ===============================
LOOKBACK_DAYS = 3            # 3 covers weekends; already-seen items are skipped anyway
READING_MINUTES = 20         # your reading budget -> 6 papers (about 18 min)
MINUTES_PER_ITEM = 3         # rough time per item
SUMMARY_MAX_CHARS = 280      # keep summaries short
MIN_SCORE = 3                # ignore weak matches
FEED_BONUS = 2               # lets blog/news items compete with the many papers
STATE_DIR = Path("state")

# Each track:
#   weight      how much you care about the topic
#   max_items   cap per track, so one busy topic can't fill the email
#   keep_one    if True, the best item survives the reading-budget trim
#   color       accent colour for the topic in the email
#   requires    the text must contain at least one of these words, otherwise the
#               track doesn't score at all (stops generic ML papers sneaking in)
#   keywords    phrases scored 3 if in the title, 1 if only in the abstract
TRACKS = {
    "Fraud detection & AML": {
        "weight": 1.3, "max_items": 2, "keep_one": True, "color": "#b42318",
        "requires": ["fraud", "laundering", "aml", "suspicious", "scam"],
        "search": ["fraud detection", "financial fraud", "money laundering", "transaction fraud"],
        "keywords": [
            "fraud detection", "financial fraud", "credit card fraud", "payment fraud",
            "transaction fraud", "bank fraud", "anti-money laundering", "money laundering",
            "suspicious transaction", "transaction monitoring", "anomaly detection",
            "graph neural network", "imbalanced", "explainab", "fraud",
        ],
    },
    "Insurance pricing": {
        "weight": 1.2, "max_items": 2, "keep_one": True, "color": "#067647",
        "requires": ["insurance", "insurer", "actuarial", "premium", "policyholder", "claim"],
        "search": ["insurance pricing", "actuarial", "insurance claims", "non-life insurance"],
        "keywords": [
            "insurance pricing", "premium", "actuarial", "claim frequency", "claim severity",
            "claims", "tweedie", "generalized linear model", "glm", "gradient boosting",
            "telematics", "mortality", "reserving", "fair pricing", "discrimination-free",
            "insurance",
        ],
    },
    "Stock & asset pricing": {
        "weight": 1.1, "max_items": 2, "keep_one": True, "color": "#1d4ed8",
        "requires": ["stock", "equit", "asset pricing", "option", "volatility",
                     "portfolio", "return", "market", "derivative"],
        "search": ["asset pricing", "stock return", "option pricing", "stock price prediction"],
        "keywords": [
            "asset pricing", "cross-section of returns", "factor model", "stock return",
            "stock price", "equity", "option pricing", "deep hedging", "volatility",
            "implied volatility", "neural sde", "black-scholes", "derivatives",
            "portfolio optimization", "market microstructure", "limit order book",
            "risk premium",
        ],
    },
    "Credit risk": {
        "weight": 1.0, "max_items": 2, "keep_one": False, "color": "#b54708",
        "requires": ["credit", "default", "loan", "borrower", "lending"],
        "search": ["credit risk", "credit scoring", "loan default", "probability of default"],
        "keywords": [
            "credit risk", "credit scoring", "default prediction", "probability of default",
            "loan default", "credit rating", "lending", "borrower", "explainab",
        ],
    },
}

ARXIV_KEYWORD_CATS = ["cs.LG", "cs.AI", "cs.CR", "stat.ML", "stat.AP", "cs.CY", "cs.SI"]
ARXIV_FINANCE_CATS = ["q-fin.CP", "q-fin.MF", "q-fin.ST", "q-fin.RM", "q-fin.PM",
                      "q-fin.PR", "q-fin.GN", "q-fin.TR"]

# Email look. Pick one: "analog" (typewriter), "paper" (editorial, cream + serif),
# "dark" (dark mode), "midnight" (white cards under a navy header).
THEME = "analog"
SANS = "-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
SERIF = "Georgia,'Times New Roman',Times,serif"
MONO = "'Courier New',Courier,'Lucida Console',monospace"
THEMES = {
    "analog": {
        "scheme": "light", "page": "#e8e2d4", "body_bg": "#f6f1e3", "body_border": "#cfc6b0",
        "header_bg": "#f6f1e3",
        "header_extra": "border:1px solid #cfc6b0;border-bottom:2px dashed #2b2622;",
        "kicker": "#b3261e", "title": "#2b2622", "sub": "#6b6257",
        "ink": "#2b2622", "text": "#3b352e", "muted": "#7a7064", "line": "#bdb39c",
        "item_rule": "1px dashed #bdb39c",
        "card_bg": "#f6f1e3", "card_border": "#cfc6b0", "cards": False, "pills": False,
        "section_size": "17px", "section_rule": "2px dashed #2b2622",
        "trend_bg": "#efe8d6", "trend_extra": "border:1px dashed #b3261e;", "trend_title": "#b3261e",
        "head_font": MONO, "body_font": MONO, "radius": "0",
        "colors": {"Fraud detection & AML": "#b3261e", "Insurance pricing": "#2e5e3e",
                   "Stock & asset pricing": "#24426e", "Credit risk": "#8a5a12"},
    },
    "paper": {
        "scheme": "light", "page": "#efe9dd", "body_bg": "#fbf8f2", "body_border": "#e2d8c6",
        "header_bg": "#fbf8f2", "header_extra": "border:1px solid #e2d8c6;border-bottom:3px double #1c1a17;",
        "kicker": "#9a3b2e", "title": "#1c1a17", "sub": "#6f655a",
        "ink": "#1c1a17", "text": "#3d372f", "muted": "#857a6c", "line": "#ddd2bf",
        "card_bg": "#fbf8f2", "card_border": "#e2d8c6", "cards": False, "pills": False,
        "section_size": "20px", "section_rule": "2px solid #1c1a17",
        "trend_bg": "#f3ede2", "trend_extra": "border-left:3px solid #9a3b2e;", "trend_title": "#9a3b2e",
        "head_font": SERIF, "body_font": SANS, "radius": "0",
        "colors": {"Fraud detection & AML": "#9a3b2e", "Insurance pricing": "#3d6b4f",
                   "Stock & asset pricing": "#2f4f7f", "Credit risk": "#93611a"},
    },
    "dark": {
        "scheme": "dark", "page": "#0b0e14", "body_bg": "#0f131a", "body_border": "#232a35",
        "header_bg": "#141a23", "header_extra": "border:1px solid #232a35;border-bottom:2px solid #58a6ff;",
        "kicker": "#58a6ff", "title": "#f0f4fa", "sub": "#8b96a5",
        "ink": "#f0f4fa", "text": "#c3ccd8", "muted": "#7d8794", "line": "#2a323e",
        "card_bg": "#161c25", "card_border": "#262e3a", "cards": True, "pills": True,
        "section_size": "15px", "section_rule": "1px solid #232a35",
        "trend_bg": "#141c2b", "trend_extra": "", "trend_title": "#58a6ff",
        "head_font": SANS, "body_font": SANS, "radius": "12px",
        "colors": {"Fraud detection & AML": "#ff7b72", "Insurance pricing": "#3fb950",
                   "Stock & asset pricing": "#58a6ff", "Credit risk": "#e3b341"},
    },
    "midnight": {
        "scheme": "light", "page": "#f2f4f7", "body_bg": "#f9fafb", "body_border": "#eaecf0",
        "header_bg": "#0b1f3a", "header_extra": "",
        "kicker": "#9db4d6", "title": "#ffffff", "sub": "#c9d6ea",
        "ink": "#101828", "text": "#344054", "muted": "#667085", "line": "#eaecf0",
        "card_bg": "#ffffff", "card_border": "#eaecf0", "cards": True, "pills": True,
        "section_size": "15px", "section_rule": "1px solid #eaecf0",
        "trend_bg": "#eef4ff", "trend_extra": "", "trend_title": "#1d4ed8",
        "head_font": SANS, "body_font": SANS, "radius": "12px",
        "colors": {},
    },
}

# Best-effort feed URLs. Dead feeds are skipped with a warning in the log; fix or remove them.
RSS_FEEDS = {
    "Quantocracy": "https://quantocracy.com/feed/",
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
        terms = track.get("search") or track["keywords"]
        kws = " OR ".join(f'all:"{k}"' for k in terms)
        queries.append(f"({cats}) AND ({kws})")

    for i, q in enumerate(queries):
        try:
            r = requests.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": q,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": 100 if i == 0 else 40,  # finance feed is busier
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
        time.sleep(3)  # arXiv asks for ~3 s between API calls
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
    best_track, best_score, hits, best_hits = None, 0.0, set(), []
    text = f"{title} {body}"
    for track, cfg in TRACKS.items():
        req = cfg.get("requires") or []
        if req and not any(w in text for w in req):
            continue  # off-topic for this track
        raw, matched = 0, []
        for kw in cfg["keywords"]:
            in_t, in_b = kw in title, kw in body
            if in_t or in_b:
                hits.add(kw)
                matched.append(kw)
                raw += 3 if in_t else 1
        s = raw * cfg["weight"]
        if s > best_score:
            best_track, best_score, best_hits = track, s, matched
    if best_score == 0:
        return None, 0, hits
    item["matched"] = best_hits
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
        picked[t] = lst[:TRACKS[t]["max_items"]]

    # trim to the reading budget, dropping the lowest scores first,
    # but never the top item of a keep_one track
    protected = {id(l[0]) for t, l in picked.items() if l and TRACKS[t]["keep_one"]}
    max_items = max(1, READING_MINUTES // MINUTES_PER_ITEM)
    flat = sorted(
        ((it["score"], t, it) for t, l in picked.items() for it in l), key=lambda x: x[0]
    )
    total = len(flat)
    for _, t, it in flat:
        if total <= max_items:
            break
        if id(it) in protected:
            continue
        picked[t].remove(it)
        total -= 1
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
        "why it matters to an AI undergraduate focused on machine learning in finance: fraud "
        "detection in banks, insurance pricing, stock/asset pricing and credit risk. "
        'Return only JSON like {"0": "sentence", "1": "sentence"}.\n\n'
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
def tint(hex_color, amount=0.1, base="#ffffff"):
    """Mix a colour into a background colour (used for the topic pills)."""
    rgb = lambda h: [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    out = [round(b + (c - b) * amount) for c, b in zip(rgb(hex_color), rgb(base))]
    return "#" + "".join(f"{v:02x}" for v in out)


def pdf_link(link):
    """arXiv abstract page -> direct PDF link (None for other sources)."""
    if "arxiv.org/abs/" in link:
        return link.replace("/abs/", "/pdf/")
    return None


def render(picked, oneliners, trend, today):
    """Email-safe HTML (tables + inline styles, works in Gmail/Outlook/phone) plus plain text."""
    T = THEMES.get(THEME, THEMES["midnight"])
    e = html.escape
    items = [(t, it) for t, lst in picked.items() for it in lst]
    total = len(items)
    minutes = total * MINUTES_PER_ITEM
    nice_date = datetime.fromisoformat(today).strftime("%A %d %B %Y")

    body_font = f"font-family:{T['body_font']};"
    head_font = f"font-family:{T['head_font']};"
    color = lambda track: T["colors"].get(track, TRACKS[track]["color"])
    radius = T["radius"]

    def label(track):
        c = color(track)
        if T["pills"]:
            return (f'<span style="display:inline-block;padding:2px 9px;border-radius:999px;'
                    f'background:{tint(c, 0.14, T["card_bg"])};color:{c};font-size:11px;'
                    f'font-weight:600;letter-spacing:.3px;text-transform:uppercase">{e(track)}</span>')
        return (f'<span style="color:{c};font-size:11px;font-weight:700;letter-spacing:1.2px;'
                f'text-transform:uppercase">{e(track)}</span>')

    def summary_of(it):
        return oneliners.get(it["link"]) or shorten(it["summary"], SUMMARY_MAX_CHARS)

    def card(track, it, big=False):
        c = color(track)
        pdf = pdf_link(it["link"])
        tags = ", ".join(it.get("matched", [])[:3])
        links = (f'<a href="{e(it["link"])}" style="color:{c};font-weight:600;'
                 f'text-decoration:none">Read abstract&nbsp;&rarr;</a>')
        if pdf:
            links += (f'&nbsp;&nbsp;<span style="color:{T["line"]}">|</span>&nbsp;&nbsp;'
                      f'<a href="{e(pdf)}" style="color:{T["muted"]};text-decoration:none">PDF</a>')
        size = "22px" if big else "17px"
        if T["cards"]:
            box = (f"background:{T['card_bg']};border:1px solid {T['card_border']};"
                   f"border-left:4px solid {c};border-radius:{radius}")
            pad, gap = "16px 18px", "14px"
        else:  # editorial: no boxes, just a rule under each item
            box = f"border-bottom:{T.get('item_rule', '1px solid ' + T['line'])}"
            pad, gap = "16px 2px 18px", "4px"
        return f"""
<tr><td style="padding:0 0 {gap} 0">
 <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="{box}">
  <tr><td style="padding:{pad};{body_font}">
   <div style="margin-bottom:8px">{label(track)}
    <span style="color:{T['muted']};font-size:12px">&nbsp;&nbsp;{e(it['source'])} &middot; {it['published']:%d %b} &middot; ~{MINUTES_PER_ITEM} min</span></div>
   <a href="{e(it['link'])}" style="color:{T['ink']};text-decoration:none;font-size:{size};
      line-height:1.3;font-weight:700;{head_font}">{e(it['title'])}</a>
   <p style="margin:8px 0 10px 0;color:{T['text']};font-size:14px;line-height:1.6">{e(summary_of(it))}</p>
   {f'<div style="color:{T["muted"]};font-size:12px;margin-bottom:10px">Why you got this: {e(tags)}</div>' if tags else ''}
   <div style="font-size:13px">{links}</div>
  </td></tr>
 </table>
</td></tr>"""

    def heading(text, c=None):
        mark = f'<span style="color:{c}">&#9632;</span>&nbsp; ' if c and T["cards"] else ""
        return (f'<tr><td style="{head_font}color:{T["ink"]};font-size:{T["section_size"]};'
                f'font-weight:700;padding:16px 0 10px;border-top:{T["section_rule"]}">'
                f'{mark}{e(text)}</td></tr>')

    # top pick = highest score overall, shown first and not repeated below
    top = max(items, key=lambda x: x[1]["score"]) if items else None

    rows = []
    if top:
        rows.append(f'<tr><td style="{body_font}color:{T["muted"]};font-size:12px;font-weight:700;'
                    f'letter-spacing:1.5px;padding:4px 0 8px">&#9733; TOP PICK TODAY</td></tr>')
        rows.append(card(*top, big=True))
    for track, lst in picked.items():
        rest = [it for it in lst if not (top and it is top[1])]
        if rest:
            rows.append(heading(track, color(track)))
            rows.extend(card(track, it) for it in rest)
    if not items:
        rows.append(f'<tr><td style="{body_font}color:{T["muted"]};padding:20px 0">'
                    f"No strong matches today. Enjoy the free 20 minutes.</td></tr>")

    covered = [t for t, l in picked.items() if l]
    top_radius = f"{radius} {radius} 0 0"
    bottom_radius = f"0 0 {radius} {radius}"
    html_body = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="{T['scheme']}"><meta name="supported-color-schemes" content="{T['scheme']}"></head>
<body style="margin:0;padding:0;background:{T['page']}">
<div style="display:none;max-height:0;overflow:hidden">{total} papers on {e(', '.join(covered).lower()) or 'ML in finance'} &middot; about {minutes} min</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{T['page']}">
<tr><td align="center" style="padding:24px 12px">
 <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:640px">
  <tr><td style="background:{T['header_bg']};border-radius:{top_radius};padding:28px 24px 24px;{T['header_extra']}">
   <div style="{body_font}color:{T['kicker']};font-size:12px;letter-spacing:2px;font-weight:700">ML IN FINANCE &middot; DAILY BRIEF</div>
   <div style="{head_font}color:{T['title']};font-size:30px;font-weight:700;margin-top:8px;line-height:1.15">{nice_date}</div>
   <div style="{body_font}color:{T['sub']};font-size:14px;margin-top:10px">{total} papers &middot; about {minutes} minutes</div>
  </td></tr>
  <tr><td style="background:{T['body_bg']};border:1px solid {T['body_border']};border-top:0;border-radius:{bottom_radius};padding:20px 20px 6px">
   <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
    {''.join(rows)}
    <tr><td style="padding:10px 0 18px">
     <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
      style="background:{T['trend_bg']};border-radius:{radius};{T['trend_extra']}"><tr><td style="padding:14px 18px;{body_font}">
      <div style="color:{T['trend_title']};font-size:12px;font-weight:700;letter-spacing:1.5px">&#8599; TRENDS</div>
      <div style="color:{T['text']};font-size:14px;line-height:1.55;margin-top:4px">{e(trend)}</div>
     </td></tr></table>
    </td></tr>
   </table>
  </td></tr>
  <tr><td style="{body_font}color:{T['muted']};font-size:12px;text-align:center;padding:16px">
   Sent by your research digest &middot; change THEME or topics in the CONFIG block of digest.py
  </td></tr>
 </table>
</td></tr></table></body></html>"""

    text = [f"ML in finance - daily brief, {nice_date}", f"{total} papers, about {minutes} min", ""]
    if top:
        text += ["TOP PICK", f"* {top[1]['title']}  [{top[0]}]", f"  {summary_of(top[1])}",
                 f"  {top[1]['link']}", ""]
    for track, lst in picked.items():
        rest = [it for it in lst if not (top and it is top[1])]
        if rest:
            text.append(f"== {track} ==")
            for it in rest:
                text += [f"- {it['title']} [{it['source']}]", f"  {summary_of(it)}", f"  {it['link']}"]
            text.append("")
    text.append(f"Trends: {trend}")
    return html_body, "\n".join(text), minutes


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
    n = sum(len(v) for v in picked.values())
    subject = f"Research digest · ML in finance · {now:%d %b} · {n} papers, ~{minutes} min"

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
