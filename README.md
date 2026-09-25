# AI Tester

**Try it:** https://ai-tester-saraissa.streamlit.app/

A black-box evaluation tool for AI agents — score, track, and audit an agent's answers from the outside, without touching its code or weights.

## About

Testing an AI agent shouldn't mean copy-pasting questions into a chat window one at a time. AI Tester lets you write or auto-generate test cases, send them to any target agent through its API, and score the answers automatically — so evaluation is scripted and repeatable instead of manual.

## Features

- **Test case management** — write your own test cases or generate them automatically
- **API-driven evaluation** — sends test cases to the target agent via its API, no manual copy-paste
- **Two scoring modes** — offline text-similarity, or an AI judge for nuanced grading
- **Bias & fairness checks** — flags skewed or inconsistent answers across similar prompts
- **Run history** — every run is saved to Postgres, so results can be tracked and compared over time
- **User accounts & projects** — multiple projects and roles, backed by Postgres (Supabase)

## Requirements

- Python 3.10+
- A Postgres database (a free [Supabase](https://supabase.com) project works) — the app won't start without a `DATABASE_URL`

## Getting Started

**1. Clone and enter the project:**
```bash
git clone https://github.com/Saraissa00/AI-Tester.git
cd AI-Tester
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Set your database connection:**

Add a `DATABASE_URL` either as an environment variable, or in `.streamlit/secrets.toml`:
```toml
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
```

**4. Run the app:**
```bash
streamlit run app.py
```

**5. Open it in your browser:**

Streamlit prints a local URL (usually `http://localhost:8501`) — open it to start testing.

## Tech Stack

- **Streamlit** — app framework and UI
- **PostgreSQL (Supabase)** — persists projects, test cases, and run history
- **REST APIs** — used to reach the AI agent under test

