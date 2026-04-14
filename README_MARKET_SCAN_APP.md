# Market Scan App

This is the first web-app version of the market scanner.

## What it does

- scans a watchlist using `yfinance`
- gates the dashboard behind a lightweight email-or-phone login
- remembers each user's watchlist and morning alert preference
- highlights the current leader
- shows market stance
- previews a 5am premarket pulse message
- shows a live-tape trust layer beside the model read
- separates:
  - best longs
  - best shorts
  - avoid
- displays the full watchlist table in a browser

## Run locally

```bash
uvicorn market_scan_app:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Delivery providers

The app can now send real test alerts and run a morning pulse batch if you set provider credentials.

### SMS with Twilio

Set:

```bash
export TWILIO_ACCOUNT_SID='your_sid'
export TWILIO_AUTH_TOKEN='your_token'
export TWILIO_FROM_PHONE='+15551234567'
```

### Email with Resend

Set:

```bash
export RESEND_API_KEY='your_resend_api_key'
export RESEND_FROM_EMAIL='Market Scanner <alerts@yourdomain.com>'
```

## Morning pulse runner

To send the pulse to every saved user with alerts enabled:

```bash
python3 '/Users/moseswilling/Documents/New project/send_morning_pulse.py'
```

For a real 5am workflow, this script should be scheduled by cron, launchd, or an automation layer later.

## Query parameters

- `symbols`
- `months`

Example:

```text
http://127.0.0.1:8000/?symbols=NVDA,AAPL,MSFT,META,TSLA,AMD,AMZN,QQQ,SPY&months=6
```

## Product direction

This is the first app-shaped layer, not the finished product. The next likely
steps are:

- actual SMS / email delivery integration
- subscription / payments
- saved watchlists
- chart detail pages
- historical scan history
