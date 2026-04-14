# Deploy Market Scan App

This app is ready for a first hosted deployment as:

- one **web service** for the FastAPI app
- one **cron job** for the morning pulse runner

## Recommended platform

Railway is the easiest first path because it supports both web services and scheduled jobs.

## Files already prepared

- `requirements.txt`
- `Procfile`
- `railway.json`
- `.env.example`

## Environment variables

Set these in the hosting platform:

```text
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_FROM_PHONE
RESEND_API_KEY
RESEND_FROM_EMAIL
MARKET_SCAN_DB_PATH
```

If you stay on SQLite for the first hosted version, use something like:

```text
MARKET_SCAN_DB_PATH=/data/market_scan_users.db
```

For a stronger production setup later, switch the user table to Postgres.

## Railway web service

Start command:

```bash
uvicorn market_scan_app:app --host 0.0.0.0 --port $PORT
```

Health check path:

```text
/login
```

## Railway cron job

Command:

```bash
python3 send_morning_pulse.py
```

Schedule:

- `0 9 * * *` for **5:00 AM EDT**
- `0 10 * * *` for **5:00 AM EST**

If you want stable local-time delivery year-round, use a platform scheduler that supports time zones or move the scheduling logic into the app.

## Publish flow

1. Push this folder to GitHub.
2. Create a Railway project from the repo.
3. Add the environment variables.
4. Create the web service.
5. Create a scheduled job that runs `python3 send_morning_pulse.py`.
6. Test login, save a phone number, and send a test pulse.

## Current caveats

- The app still uses SQLite by default.
- Email delivery requires Resend setup.
- SMS delivery depends on Twilio account status and messaging approval.
- The current login is a lightweight identifier flow, not full OTP verification.
