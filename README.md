# AI Tester

**Live:** https://ai-tester-saraissa.streamlit.app/

A black-box evaluation tool for AI agents — write or generate test cases, send them to an agent via API, score the answers (offline text-similarity or an AI judge), check for bias, and track results over time with saved run history. Built with Streamlit, data persisted in Postgres (Supabase).

## Running it locally

```
pip install -r requirements.txt
streamlit run app.py
```

Needs a `DATABASE_URL` (Postgres) in `.streamlit/secrets.toml` or as an environment variable to persist data — without it the app won't start.
