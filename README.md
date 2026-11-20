# Computational finance digest

A daily email with about 6 new research papers (a 20-minute read) on machine learning in finance:

- **Fraud detection & AML**: card and payment fraud, anti-money laundering, transaction monitoring
- **Insurance pricing**: actuarial models, claims, GLMs vs. machine learning, fair pricing
- **Stock & asset pricing**: asset pricing, factor models, option pricing, volatility
- **Credit risk**: credit scoring, probability of default, loan default

Papers come from arXiv (finance and machine learning sections), Hugging Face Daily Papers and two finance feeds (Quantocracy, BIS). Each paper is scored against keywords for these topics, and the email is sent to your Gmail in a plain, simple layout.

## Setup

1. Create a GitHub repo and upload everything in this folder, keeping the `.github/workflows/digest.yml` path.
2. Create a Gmail App Password: turn on 2-step verification, then go to Google Account > Security > App passwords.
3. Add repo secrets (Settings > Secrets and variables > Actions > New repository secret):
   - `GMAIL_ADDRESS`: your Gmail address
   - `GMAIL_APP_PASSWORD`: the 16-character app password
   - `TO_EMAIL`: where to send the digest (usually the same Gmail address)
   - `ANTHROPIC_API_KEY`: optional. If set, each paper gets a one-sentence "why it matters" instead of an abstract excerpt.
4. Run it once: Actions tab > "Daily research digest" > Run workflow.

After that it runs every day at 05:00 UTC. GitHub often starts scheduled runs late, sometimes by several hours.

Tip: the subject starts with "Computational finance", so you can make a Gmail filter on that text to label the emails.

## Test locally

```
pip install -r requirements.txt
DRY_RUN=1 python digest.py     # prints the digest, sends nothing
```

## Settings

Everything is in the CONFIG block at the top of `digest.py`:

- `READING_MINUTES`: your reading budget. Papers per email = this divided by `MINUTES_PER_ITEM` (20 / 3 = 6).
- `TRACKS`: the topics. Each has:
  - `weight`: how much you care about it
  - `max_items`: the most papers from this topic per email
  - `keep_one`: always keep the best paper from this topic, even if the email is full
  - `requires`: a paper must mention one of these words to count for the topic
  - `search`: phrases used to search arXiv
  - `keywords`: phrases used to score papers (3 points in the title, 1 in the abstract)
- `RSS_FEEDS`: extra sources. Dead feeds print a warning in the Actions log and are skipped.

## Files

- `digest.py`: the script
- `.github/workflows/digest.yml`: the daily schedule
- `state/`: created automatically. `seen.json` stops papers being sent twice, and `trends.json` holds keyword counts for the trend line. Committed by the workflow.
