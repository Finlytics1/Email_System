# Research digest

A daily email with about 8 items (10-30 min read) on AI, computational finance, privacy, and your bachelor project (membership inference and neural-network option pricing). It is sent to your main Gmail.

## Setup (about 10 minutes)

1. **Create a GitHub repo** (private is fine) and upload everything in this folder, keeping the `.github/workflows/digest.yml` path.
2. **Create a Gmail App Password** for your main Gmail. Turn on 2-step verification first, then go to Google Account > Security > App passwords. The digest is sent from and to the same address, so no second account is needed.
3. **Add repo secrets** (Settings > Secrets and variables > Actions > New repository secret):
   - `GMAIL_ADDRESS`: your main Gmail address
   - `GMAIL_APP_PASSWORD`: the 16-character app password
   - `TO_EMAIL`: your main Gmail address again
   - `ANTHROPIC_API_KEY`: *optional*. If set, each item gets a one-sentence "why it matters" instead of an abstract excerpt.
4. **Run it once manually:** Actions tab > "Daily research digest" > Run workflow.
5. **Check your inbox.** Because it is sent to yourself, it should land in the inbox. If not, look in Spam and mark it "not spam". You can also create a filter that adds a "Digest" label to emails with the subject "Research digest".

After that it runs every day at 05:00 UTC (06:00 or 07:00 Danish time). GitHub can delay scheduled runs by up to about an hour.

## Test locally

```bash
pip install -r requirements.txt
DRY_RUN=1 python digest.py     # prints the digest, sends nothing
```

## Tuning

Everything is in the `CONFIG` block at the top of `digest.py`:

- `READING_MINUTES`: your budget. Items per email is this divided by `MINUTES_PER_ITEM`.
- `ITEMS_PER_TRACK`: max items per track.
- `TRACKS`: keywords and weights. The project track has weight 2.0.
- `RSS_FEEDS`: add or remove news sources. Dead feeds print a warning in the Actions log and are skipped.

## Files

- `digest.py`: the script
- `.github/workflows/digest.yml`: daily schedule
- `state/`: created automatically. Holds `seen.json` (avoids repeats) and `trends.json` (keyword counts for the trend line). Committed by the workflow.
