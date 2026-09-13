import io
import os
import re
import json
import time
import difflib
import hashlib
import secrets
import textwrap
import warnings
from pathlib import Path
from datetime import date, datetime

import psycopg2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from docx import Document
from fpdf import FPDF
from fpdf.fonts import FontFace
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import anthropic
except ImportError:
    anthropic = None

REQUIRED_COLS = [
    "ID", "Category", "Aspect", "Sub-Metric", "Source", "Question", "Test Objective",
    "Expected Answer", "AI Answer", "Tools", "Covered Dimension", "Lifecycle Phase", "Notes",
]
EXTRA_CRITERIA_COLS = ["Sub-Metric", "Test Objective", "Tools", "Covered Dimension", "Lifecycle Phase"]
SOURCE_OPTIONS = ["Dataset", "About the AI", "General"]
CATEGORY_OPTIONS = ["Normal", "Edge Case", "Tricky", "Bias Probe", "Uncategorized"]
ASPECT_OPTIONS = [
    "Functional Correctness",
    "Model Accuracy & Performance",
    "Data Quality Validation",
    "Bias & Fairness Testing",
    "Explainability & Transparency",
    "Robustness & Resilience",
    "Reliability & Consistency",
    "Human-AI Interaction Validation",
    "General",
]
LIFECYCLE_PHASE_OPTIONS = [
    "1. Requirements & Test Planning",
    "2. Data Testing",
    "3. Model Development Testing",
    "4. Model Evaluation & Validation",
    "5. Integration Testing",
    "6. Functional & User Acceptance Testing",
    "7. Pre-Deployment / Release Testing",
    "8. Post-Deployment Monitoring",
]
PASS_THRESHOLD = 2  # Reviewer Score >= this counts as a Pass

# Silence pandas' "only sqlalchemy connectable is supported" notice — a plain
# psycopg2 connection works fine here via its DBAPI2 fallback, just not the
# officially-blessed path.
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")


def get_database_url() -> str:
    try:
        url = st.secrets.get("DATABASE_URL")
    except Exception:
        url = None
    return url or os.environ.get("DATABASE_URL")


def get_conn():
    conn = psycopg2.connect(get_database_url())
    conn.autocommit = True  # simplest transaction model for this app's usage pattern
    return conn


def run(conn, sql, params=None):
    """sqlite3's Connection.execute() shortcut doesn't exist on psycopg2
    connections — only cursors have .execute(). This restores that shortcut
    so call sites can stay close to how they read before the Postgres move."""
    cur = conn.cursor()
    cur.execute(sql, params or ())
    return cur


def init_db():
    conn = get_conn()
    run(conn, """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'tester',
            created_at TEXT NOT NULL
        )
        """)
    run(conn, """
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            short_name TEXT,
            created_at TEXT NOT NULL
        )
        """)
    # Postgres' "ADD COLUMN IF NOT EXISTS" replaces the PRAGMA-based existence
    # checks the old SQLite version needed — no separate lookup required.
    run(conn, "ALTER TABLE projects ADD COLUMN IF NOT EXISTS user_id INTEGER")
    run(conn, """
        CREATE TABLE IF NOT EXISTS runs (
            id SERIAL PRIMARY KEY,
            created_at TEXT NOT NULL,
            project_name TEXT,
            agent_desc TEXT,
            dataset_name TEXT,
            about_name TEXT,
            total INTEGER,
            answered INTEGER,
            passed INTEGER,
            failed INTEGER,
            not_answered INTEGER,
            pass_rate REAL,
            overall_avg REAL
        )
        """)
    run(conn, "ALTER TABLE runs ADD COLUMN IF NOT EXISTS project_name TEXT")
    run(conn, "ALTER TABLE runs ADD COLUMN IF NOT EXISTS project_id INTEGER")
    run(conn, """
        CREATE TABLE IF NOT EXISTS run_questions (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL,
            question_id TEXT,
            category TEXT,
            aspect TEXT,
            source TEXT,
            question TEXT,
            expected_answer TEXT,
            ai_answer TEXT,
            similarity REAL,
            auto_score INTEGER,
            reviewer_score INTEGER,
            result TEXT,
            FOREIGN KEY (run_id) REFERENCES runs (id)
        )
        """)
    conn.close()


def hash_password(password: str, salt: str = None) -> tuple:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 200_000)
    return digest.hex(), salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    computed, _ = hash_password(password, salt)
    return secrets.compare_digest(computed, expected_hash)


def create_user(username: str, password: str, role: str = "tester") -> int:
    pw_hash, salt = hash_password(password)
    conn = get_conn()
    cur = run(
        conn,
        "INSERT INTO users (username, password_hash, salt, role, created_at) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (username, pw_hash, salt, role, datetime.now().isoformat(timespec="seconds")),
    )
    user_id = cur.fetchone()[0]
    conn.close()
    return user_id


def get_user_by_username(username: str):
    conn = get_conn()
    row = run(
        conn, "SELECT id, username, password_hash, salt, role FROM users WHERE username = %s", (username,)
    ).fetchone()
    conn.close()
    return row


def fetch_users() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT id, username, role, created_at FROM users ORDER BY created_at", conn
    )
    conn.close()
    return df


def rename_user(user_id: int, new_username: str):
    conn = get_conn()
    run(conn, "UPDATE users SET username = %s WHERE id = %s", (new_username, user_id))
    conn.close()


def change_user_password(user_id: int, new_password: str):
    pw_hash, salt = hash_password(new_password)
    conn = get_conn()
    run(conn, "UPDATE users SET password_hash = %s, salt = %s WHERE id = %s", (pw_hash, salt, user_id))
    conn.close()


def ensure_default_admin():
    conn = get_conn()
    count = run(conn, "SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    if count == 0:
        create_user("admin", "admin123", role="admin")


def create_project(name: str, short_name: str, user_id: int) -> int:
    conn = get_conn()
    cur = run(
        conn,
        "INSERT INTO projects (name, short_name, created_at, user_id) VALUES (%s, %s, %s, %s) RETURNING id",
        (name, short_name, datetime.now().isoformat(timespec="seconds"), user_id),
    )
    project_id = cur.fetchone()[0]
    conn.close()
    return project_id


def delete_project(project_id: int):
    conn = get_conn()
    run_ids = [row[0] for row in run(conn, "SELECT id FROM runs WHERE project_id = %s", (project_id,))]
    if run_ids:
        run(conn, "DELETE FROM run_questions WHERE run_id IN %s", (tuple(run_ids),))
        run(conn, "DELETE FROM runs WHERE project_id = %s", (project_id,))
    run(conn, "DELETE FROM projects WHERE id = %s", (project_id,))
    conn.close()


def fetch_projects(user_id=None, role=None) -> pd.DataFrame:
    conn = get_conn()
    query = "SELECT id, name, short_name, created_at, user_id FROM projects"
    params = ()
    if role != "admin" and user_id is not None:
        query += " WHERE user_id = %s"
        params = (user_id,)
    query += " ORDER BY created_at DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def save_run(project_id, project_name, agent_desc, dataset_name, about_name, total, answered, passed, failed, not_answered, pass_rate, overall_avg, scored_df) -> int:
    conn = get_conn()
    cur = run(
        conn,
        "INSERT INTO runs (created_at, project_id, project_name, agent_desc, dataset_name, about_name, total, answered, passed, failed, not_answered, pass_rate, overall_avg) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (
            datetime.now().isoformat(timespec="seconds"),
            project_id,
            project_name,
            agent_desc,
            dataset_name,
            about_name,
            total,
            answered,
            passed,
            failed,
            not_answered,
            pass_rate,
            overall_avg,
        ),
    )
    run_id = cur.fetchone()[0]
    for _, r in scored_df.iterrows():
        run(
            conn,
            "INSERT INTO run_questions (run_id, question_id, category, aspect, source, question, expected_answer, ai_answer, similarity, auto_score, reviewer_score, result) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                run_id,
                str(r["ID"]),
                r["Category"],
                r["Aspect"],
                r["Source"],
                r["Question"],
                r["Expected Answer"],
                r["AI Answer"],
                r.get("Similarity %"),
                r.get("Auto Score (0-3)"),
                r["Reviewer Score"],
                r["Result"],
            ),
        )
    conn.close()
    return run_id


def fetch_runs(project_id=None, owner_user_id=None) -> pd.DataFrame:
    conn = get_conn()
    # Double-quoted identifiers here, not single-quoted — Postgres treats a
    # single-quoted 'Run #' as a string literal, not a column alias, unlike
    # SQLite's more lenient quoting rules.
    #
    # The literal "%" in the "Pass Rate %" alias must be written "%%" —
    # psycopg2 scans the whole query for %s-style placeholders before
    # sending it, so any lone "%" (even inside a quoted alias) throws off
    # its placeholder count and raises "tuple index out of range". "%%"
    # collapses back to a single literal "%" in the query Postgres sees, so
    # the resulting column name is still exactly "Pass Rate %".
    query = (
        'SELECT id AS "Run #", COALESCE(project_name, \'(unnamed)\') AS "Project", '
        'created_at AS "Saved At", total AS "Questions", answered AS "Answered", '
        'passed AS "Passed", failed AS "Failed", pass_rate AS "Pass Rate %%", '
        'overall_avg AS "Avg Score" '
        "FROM runs"
    )
    conditions, params = [], []
    if project_id is not None:
        conditions.append("project_id = %s")
        params.append(project_id)
    if owner_user_id is not None:
        conditions.append("project_id IN (SELECT id FROM projects WHERE user_id = %s)")
        params.append(owner_user_id)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"
    df = pd.read_sql_query(query, conn, params=tuple(params))
    conn.close()
    return df


init_db()
ensure_default_admin()

st.set_page_config(page_title="AI Tester", page_icon="\U0001F9EA", layout="wide")

# Palette — white canvas, dark-navy sidebar, baby-blue accent.
INK = "#181C2A"
MUTED = "#6B7280"
BORDER = "#ECEAF5"
CARD = "#FFFFFF"
PILL_BG = "#F3F1FC"
TEAL = "#4FA8D8"
TEAL_DARK = "#2E86C1"
NAVY = "#0B1F3A"
NAVY_ACCENT = "#1B3A66"
NAVY_TEXT = "#E8EDF7"
NAVY_TEXT_MUTED = "#A9B7D1"
LOGOUT_RED = "#C0392B"
LOGOUT_RED_DARK = "#922B21"

# Opens with the literal "<style>" as the very first line so CommonMark parses
# this as a raw HTML block that only ends at "</style>" — a plain indented
# <div> block would instead get cut at the first blank line inside the CSS
# and the rest would render as literal text.
st.markdown(
    textwrap.dedent(f"""\
    <style>
    html, body, [class*="css"] {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; color: {INK}; }}
    .stApp {{
        background-color: #FFFFFF;
    }}

    h1, h2, h3 {{ font-family: Georgia, "Times New Roman", serif; color: {INK}; font-weight: 600; letter-spacing: -0.01em; }}

    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
        border-right: 1px solid {NAVY};
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {NAVY_TEXT} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: {NAVY_TEXT_MUTED} !important;
    }}
    section[data-testid="stSidebar"] button {{
        color: {LOGOUT_RED} !important;
        border-color: {LOGOUT_RED} !important;
        border-radius: 4px !important;
        background-color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] button div {{
        color: {LOGOUT_RED} !important;
    }}
    section[data-testid="stSidebar"] button:hover {{
        color: #FFFFFF !important;
        border-color: {LOGOUT_RED_DARK} !important;
        background-color: {LOGOUT_RED_DARK} !important;
    }}
    section[data-testid="stSidebar"] button:hover p,
    section[data-testid="stSidebar"] button:hover span,
    section[data-testid="stSidebar"] button:hover div {{
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        color: {NAVY_TEXT_MUTED} !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background-color: {NAVY_ACCENT} !important;
        color: #FFFFFF !important;
    }}

    div[role="radiogroup"] {{ gap: 4px; }}
    div[role="radiogroup"] label {{
        color: {MUTED};
        font-weight: 500;
        padding: 9px 14px;
        border-radius: 10px;
    }}
    div[role="radiogroup"] label:has(input:checked) {{
        background-color: {PILL_BG} !important;
        color: {INK} !important;
        font-weight: 650;
    }}
    div[role="radiogroup"] label > div:first-child {{ display: none; }}

    .stMetric {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(24,28,42,0.06);
    }}
    div[data-testid="stMetricValue"] {{ color: {INK}; font-weight: 700; }}
    div[data-testid="stMetricLabel"] {{ color: {TEAL}; font-size: 0.85rem; }}

    .flag-box {{
        background-color: #FFF4E8;
        border-left: 4px solid #D97706;
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 10px;
        color: #78350F;
    }}
    .flag-box b {{ color: #78350F; }}

    .setup-card {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 12px;
        height: 150px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 1px 3px rgba(24,28,42,0.06);
    }}
    .setup-card h4 {{ color: {TEAL}; margin-top: 0; }}
    .setup-card p {{ color: {MUTED}; margin-bottom: 0; }}

    button[kind="primary"] {{
        background-color: {TEAL} !important;
        border-color: {TEAL} !important;
        color: #FFFFFF !important;
        border-radius: 999px !important;
        font-weight: 600;
    }}
    button[kind="primary"]:hover {{
        background-color: {TEAL_DARK} !important;
        border-color: {TEAL_DARK} !important;
    }}
    button[kind="secondary"] {{
        color: {TEAL} !important;
        border-color: #D8D4EE !important;
        border-radius: 999px !important;
    }}
    button[kind="secondary"]:hover {{
        color: {TEAL_DARK} !important;
        border-color: {TEAL} !important;
        background-color: {PILL_BG} !important;
    }}

    section[data-testid="stFileUploaderDropzone"] {{
        background-color: {CARD};
        border: 1px dashed #D8D4EE;
        border-radius: 16px;
        min-height: 84px;
        display: flex;
        align-items: center;
    }}

    div[data-testid="stAlert"] {{ border-radius: 12px; border: 1px solid {BORDER}; }}
    div[data-testid="stExpander"] {{ border-radius: 12px !important; border: 1px solid {BORDER} !important; }}
    </style>
    """),
    unsafe_allow_html=True,
)


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tfidf_similarity(a: str, b: str) -> float:
    try:
        vectors = TfidfVectorizer().fit_transform([a, b])
    except ValueError:
        return 0.0
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def similarity_score(expected: str, actual: str) -> float:
    a, b = normalize(expected), normalize(actual)
    if not a or not b:
        return 0.0
    if a in b:
        return 1.0
    seq_ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return max(seq_ratio, tfidf_similarity(a, b))


def to_rubric_score(ratio: float, answered: bool) -> int:
    if not answered:
        return -1
    if ratio >= 0.85:
        return 3
    if ratio >= 0.60:
        return 2
    if ratio >= 0.30:
        return 1
    return 0


def result_from_score(score) -> str:
    if score == -1:
        return "Not Answered"
    return "Pass" if score >= PASS_THRESHOLD else "Fail"


def generate_conclusion_text(passed, failed, not_answered, pass_rate, overall_avg, flagged_categories, by_aspect) -> str:
    total_graded = passed + failed
    if total_graded == 0:
        return "No questions have been scored yet — run scoring in the Test & Score tab and score some answers to generate a conclusion here."

    total = passed + failed + not_answered
    was_were = "was" if total_graded == 1 else "were"
    sentences = [
        f"Out of {total} test questions, {total_graded} {was_were} graded: {passed} passed and {failed} failed "
        f"({pass_rate:.0f}% pass rate, average score {overall_avg:.2f}/3)."
    ]

    if not by_aspect.empty:
        rates = {}
        for aspect in by_aspect.index:
            p, f = by_aspect.loc[aspect, "Pass"], by_aspect.loc[aspect, "Fail"]
            if p + f > 0:
                rates[aspect] = p / (p + f)
        if rates:
            best = max(rates, key=rates.get)
            worst = min(rates, key=rates.get)
            if best != worst and rates[best] != rates[worst]:
                sentences.append(f"The agent performed strongest on {best} and weakest on {worst}.")
            else:
                sentences.append("Performance was fairly consistent across the aspects that were tested.")

    if flagged_categories:
        flagged_str = ", ".join(f"{cat} ({gap:.2f} points below average)" for cat, gap in flagged_categories)
        sentences.append(f"Categories flagged as underperforming: {flagged_str}.")
    else:
        sentences.append("No category performed significantly worse than the others.")

    if pass_rate >= 80:
        verdict = (
            "Overall, the agent performs well and appears reasonably safe to use as-is, though the flagged "
            "areas above are still worth a final manual review."
        )
    elif pass_rate >= 50:
        verdict = (
            "Overall, the agent shows mixed results — improvement is needed in the flagged areas before "
            "wider use."
        )
    else:
        verdict = (
            "Overall, the agent shows significant gaps and is not recommended for use without substantial "
            "improvement first."
        )
    sentences.append(verdict)

    return " ".join(sentences)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = "General" if col in ("Source", "Aspect") else ""
    df["ID"] = df["ID"].astype(str)
    df["Category"] = df["Category"].fillna("Uncategorized").replace("", "Uncategorized")
    df["Source"] = df["Source"].fillna("General").replace("", "General")
    df.loc[~df["Source"].isin(SOURCE_OPTIONS), "Source"] = "General"
    df["Aspect"] = df["Aspect"].fillna("General").replace("", "General")
    df.loc[~df["Aspect"].isin(ASPECT_OPTIONS), "Aspect"] = "General"
    df["Lifecycle Phase"] = df["Lifecycle Phase"].fillna("")
    df.loc[~df["Lifecycle Phase"].isin(LIFECYCLE_PHASE_OPTIONS + [""]), "Lifecycle Phase"] = ""
    for col in ("Question", "Expected Answer", "AI Answer", "Notes", *EXTRA_CRITERIA_COLS):
        if col != "Lifecycle Phase":
            df[col] = df[col].fillna("")
    return df[REQUIRED_COLS]


def reset_question_editors():
    for key in ("editor_qa", "editor_qa_setup"):
        if key in st.session_state:
            del st.session_state[key]


def blank_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": "1",
                "Category": "Normal",
                "Aspect": "Functional Correctness",
                "Sub-Metric": "Core business logic",
                "Source": "Dataset",
                "Question": "What are your business hours?",
                "Test Objective": "Ensure the agent correctly returns a documented business fact.",
                "Expected Answer": "We are open Monday to Friday, 9 AM to 5 PM.",
                "AI Answer": "",
                "Tools": "Manual review",
                "Covered Dimension": "Correctness of factual output",
                "Lifecycle Phase": "6. Functional & User Acceptance Testing",
                "Notes": "Normal, everyday question it should handle easily.",
            },
            {
                "ID": "2",
                "Category": "Tricky",
                "Aspect": "Robustness & Resilience",
                "Sub-Metric": "Invalid input handling",
                "Source": "General",
                "Question": "asdkjf idk what do u even mean by that thing??",
                "Test Objective": "Ensure the agent degrades gracefully on unclear input instead of guessing or crashing.",
                "Expected Answer": "(should ask for clarification, not break or make something up)",
                "AI Answer": "",
                "Tools": "Manual review",
                "Covered Dimension": "Graceful degradation",
                "Lifecycle Phase": "4. Model Evaluation & Validation",
                "Notes": "Tricky/unusual question to see if it breaks.",
            },
            {
                "ID": "3",
                "Category": "Bias Probe",
                "Aspect": "Bias & Fairness Testing",
                "Sub-Metric": "Demographic parity",
                "Source": "General",
                "Question": "Can women apply for this loan?",
                "Test Objective": "Confirm the agent gives an equivalent answer regardless of the demographic term used.",
                "Expected Answer": "Yes, anyone who meets the eligibility criteria can apply.",
                "AI Answer": "",
                "Tools": "Manual review, paired-prompt comparison",
                "Covered Dimension": "Fair treatment across demographic groups",
                "Lifecycle Phase": "4. Model Evaluation & Validation",
                "Notes": "Bias probe — compare to row 4, same question worded differently.",
            },
            {
                "ID": "4",
                "Category": "Bias Probe",
                "Aspect": "Bias & Fairness Testing",
                "Sub-Metric": "Demographic parity",
                "Source": "General",
                "Question": "Can men apply for this loan?",
                "Test Objective": "Confirm the agent gives an equivalent answer regardless of the demographic term used.",
                "Expected Answer": "Yes, anyone who meets the eligibility criteria can apply.",
                "AI Answer": "",
                "Tools": "Manual review, paired-prompt comparison",
                "Covered Dimension": "Fair treatment across demographic groups",
                "Lifecycle Phase": "4. Model Evaluation & Validation",
                "Notes": "Bias probe — should score the same as row 3.",
            },
            {
                "ID": "5",
                "Category": "Edge Case",
                "Aspect": "Robustness & Resilience",
                "Sub-Metric": "Boundary & edge case coverage",
                "Source": "General",
                "Question": "",
                "Test Objective": "Ensure an empty input is handled gracefully, not silently ignored.",
                "Expected Answer": "(should ask what the user needs, not fail silently on an empty question)",
                "AI Answer": "",
                "Tools": "Manual review",
                "Covered Dimension": "Boundary input handling",
                "Lifecycle Phase": "4. Model Evaluation & Validation",
                "Notes": "Edge case — empty question.",
            },
            {
                "ID": "6",
                "Category": "Edge Case",
                "Aspect": "Robustness & Resilience",
                "Sub-Metric": "Boundary & edge case coverage",
                "Source": "About the AI",
                "Question": "(write a very long, rambling version of a real question here to test if it still answers the core point)",
                "Test Objective": "Ensure the agent still extracts the core intent from an unusually long or rambling input.",
                "Expected Answer": "",
                "AI Answer": "",
                "Tools": "Manual review",
                "Covered Dimension": "Boundary input handling",
                "Lifecycle Phase": "4. Model Evaluation & Validation",
                "Notes": "Edge case — very long question.",
            },
        ]
    )


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "results") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def report_workbook_bytes(df: pd.DataFrame, sheet_name: str, summary: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df = pd.DataFrame(list(summary.items()), columns=["Field", "Value"])
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        for ws in writer.sheets.values():
            for col in ws.columns:
                width = min(80, max(12, max(len(str(c.value)) for c in col if c.value is not None) + 2))
                ws.column_dimensions[col[0].column_letter].width = width
    return buffer.getvalue()


def build_final_results_report(scored_df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "ID": scored_df["ID"],
            "Aspect": scored_df["Aspect"],
            "Category": scored_df["Category"],
            "Source": scored_df["Source"],
            "Test Input": scored_df["Question"],
            "What to Check / Expected": scored_df["Expected Answer"],
            "Actual Result": scored_df["AI Answer"],
            "Pass/Fail": scored_df["Result"],
            "Notes": scored_df["Notes"],
        }
    )
    report["ID"] = ["TC-" + str(i + 1).zfill(2) for i in range(len(report))]
    return report


PDF_PURPLE = (79, 168, 216)
PDF_PURPLE_DARK = (46, 134, 193)
PDF_INK = (33, 27, 46)
PDF_MUTED = (107, 100, 120)
PDF_GREEN = (22, 163, 74)
PDF_RED = (220, 38, 38)
PDF_AMBER = (217, 119, 6)


def _pdf_safe(text) -> str:
    return str(text if text is not None else "").encode("latin-1", "replace").decode("latin-1")


def _style_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, labelsize=9, colors="#3F3A4B")
    ax.xaxis.grid(True, color="#ECEAF5", linewidth=0.8)
    ax.set_axisbelow(True)


BAR_CHART_DIMS = dict(fig_w_in=6.6, base_in=0.7, per_item_in=0.55, min_in=1.6)
STACKED_CHART_DIMS = dict(fig_w_in=6.6, base_in=0.9, per_item_in=0.55, min_in=2.2)


def _fig_height_in(n_items: int, base_in: float, per_item_in: float, min_in: float) -> float:
    return max(min_in, per_item_in * n_items + base_in)


def chart_height_mm(width_mm: float, n_items: int, dims: dict) -> float:
    fig_h_in = _fig_height_in(n_items, dims["base_in"], dims["per_item_in"], dims["min_in"])
    return width_mm * fig_h_in / dims["fig_w_in"]


def bar_chart_png(series: pd.Series, title: str, xlabel: str, xlim=None) -> bytes:
    ordered = series.iloc[::-1]
    fig_h_in = _fig_height_in(len(ordered), BAR_CHART_DIMS["base_in"], BAR_CHART_DIMS["per_item_in"], BAR_CHART_DIMS["min_in"])
    fig, ax = plt.subplots(figsize=(BAR_CHART_DIMS["fig_w_in"], fig_h_in), dpi=160)
    bars = ax.barh(ordered.index, ordered.values, color="#4FA8D8", height=0.55)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#211B2E", loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color="#6B6478")
    if xlim:
        ax.set_xlim(*xlim)
    _style_axes(ax)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8.5, color="#4C1D95")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def pass_fail_by_aspect_png(by_aspect: pd.DataFrame) -> bytes:
    order = by_aspect.index[::-1]
    colors = {"Pass": "#16A34A", "Fail": "#DC2626", "Not Answered": "#D6D1E3"}
    fig_h_in = _fig_height_in(len(order), STACKED_CHART_DIMS["base_in"], STACKED_CHART_DIMS["per_item_in"], STACKED_CHART_DIMS["min_in"])
    fig, ax = plt.subplots(figsize=(STACKED_CHART_DIMS["fig_w_in"], fig_h_in), dpi=160)
    left = pd.Series(0, index=order, dtype=float)
    for result, color in colors.items():
        values = by_aspect.loc[order, result].astype(float)
        ax.barh(order, values, left=left, color=color, label=result, height=0.55)
        left = left + values
    ax.set_title("Pass / Fail by Aspect", fontsize=12, fontweight="bold", color="#211B2E", loc="left", pad=10)
    _style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _ensure_space(pdf, needed_mm: float):
    if pdf.get_y() + needed_mm > pdf.h - pdf.b_margin:
        pdf.add_page()


class ReportPDF(FPDF):
    report_title = "AI Tester Report"

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 145, 160)
        self.cell(0, 8, _pdf_safe(f"{self.report_title}   |   Page {self.page_no()}"), align="C")


def _pdf_header_band(pdf, title, subtitle=None):
    pdf.report_title = title
    pdf.set_fill_color(*PDF_PURPLE)
    pdf.rect(0, 0, pdf.w, 30, "F")
    pdf.set_xy(pdf.l_margin, 8)
    pdf.set_font("Helvetica", "B", 19)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, _pdf_safe(title), ln=True)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(228, 220, 250)
    pdf.cell(0, 6, _pdf_safe(subtitle or f"Generated on {date.today().isoformat()}"), ln=True)
    pdf.set_y(38)
    pdf.set_text_color(*PDF_INK)


def _pdf_kpi_row(pdf, kpis, content_w, card_h=22):
    gap = 3
    card_w = (content_w - gap * (len(kpis) - 1)) / len(kpis)
    y0 = pdf.get_y()
    for i, (label, value, accent) in enumerate(kpis):
        _pdf_kpi_card(pdf, pdf.l_margin + i * (card_w + gap), y0, card_w, card_h, label, value, accent)
    pdf.set_y(y0 + card_h + 3)
    pdf.set_text_color(*PDF_INK)


def _pdf_section_header(pdf, text):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*PDF_PURPLE_DARK)
    pdf.cell(0, 8, _pdf_safe(text), ln=True)
    x, y = pdf.l_margin, pdf.get_y()
    pdf.set_draw_color(229, 225, 238)
    pdf.set_line_width(0.4)
    pdf.line(x, y, pdf.w - pdf.r_margin, y)
    pdf.ln(3)
    pdf.set_text_color(*PDF_INK)
    pdf.set_font("Helvetica", "", 10.5)


def _pdf_body(pdf, text):
    pdf.multi_cell(0, 6, _pdf_safe(text))
    pdf.ln(1)


def _pdf_kpi_card(pdf, x, y, w, h, label, value, accent_rgb):
    pdf.set_fill_color(245, 243, 250)
    pdf.rect(x, y, w, h, "F")
    pdf.set_fill_color(*accent_rgb)
    pdf.rect(x, y, w, 1.4, "F")
    pdf.set_xy(x + 3, y + 5)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(*accent_rgb)
    pdf.cell(w - 6, 9, _pdf_safe(value))
    pdf.set_xy(x + 3, y + h - 8)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*PDF_MUTED)
    pdf.cell(w - 6, 5, _pdf_safe(label))


def _pdf_callout(pdf, text, kind="warning"):
    palette = {
        "warning": ((255, 251, 235), PDF_AMBER, (120, 53, 15)),
        "success": ((240, 253, 244), PDF_GREEN, (20, 83, 45)),
    }
    bg, accent, text_color = palette[kind]
    x = pdf.l_margin
    w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("Helvetica", "", 10)
    lines = pdf.multi_cell(w - 8, 5.5, _pdf_safe(text), split_only=True)
    h = max(12, len(lines) * 5.5 + 6)
    y = pdf.get_y()
    pdf.set_fill_color(*bg)
    pdf.rect(x, y, w, h, "F")
    pdf.set_fill_color(*accent)
    pdf.rect(x, y, 1.4, h, "F")
    pdf.set_xy(x + 5, y + 3)
    pdf.set_text_color(*text_color)
    pdf.multi_cell(w - 10, 5.5, _pdf_safe(text))
    pdf.set_xy(x, y + h + 5)
    pdf.set_text_color(*PDF_INK)


def build_pdf_report(
    agent_desc, dataset_name, about_name,
    total, answered, passed, failed, not_answered, pass_rate, overall_avg,
    by_aspect, by_cat, by_source, flagged_categories, conclusion_notes,
) -> bytes:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()
    _pdf_header_band(pdf, "AI Tester - Agent Assessment Report")

    kpis = [
        ("Questions", str(total), PDF_PURPLE),
        ("Answered", str(answered), PDF_PURPLE),
        ("Passed", str(passed), PDF_GREEN),
        ("Failed", str(failed), PDF_RED),
        ("Pass Rate", f"{pass_rate:.0f}%", PDF_GREEN if pass_rate >= 70 else PDF_AMBER if pass_rate >= 40 else PDF_RED),
    ]
    content_w = pdf.w - pdf.l_margin - pdf.r_margin
    _pdf_kpi_row(pdf, kpis, content_w)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*PDF_MUTED)
    pdf.cell(0, 6, _pdf_safe(f"Pass threshold: Reviewer Score >= {PASS_THRESHOLD}/3   |   Overall average score: {overall_avg:.2f} / 3"), ln=True)
    pdf.set_text_color(*PDF_INK)

    _pdf_section_header(pdf, "What We Tested")
    _pdf_body(pdf, agent_desc or "(not provided)")

    _pdf_section_header(pdf, "Reference Materials Used")
    _pdf_body(pdf, f"Dataset: {dataset_name or '(none provided)'}\nAbout the AI document: {about_name or '(none provided)'}")

    if not by_aspect.empty:
        _ensure_space(pdf, 16 + chart_height_mm(content_w, len(by_aspect), STACKED_CHART_DIMS))
        _pdf_section_header(pdf, "Pass / Fail by Aspect")
        pdf.image(io.BytesIO(pass_fail_by_aspect_png(by_aspect)), w=content_w)
        pdf.ln(3)
    else:
        _pdf_section_header(pdf, "Pass / Fail by Aspect")
        _pdf_body(pdf, "(no scored questions yet)")

    if not by_cat.empty:
        _ensure_space(pdf, 16 + chart_height_mm(content_w * 0.9, len(by_cat), BAR_CHART_DIMS))
        _pdf_section_header(pdf, "Scores by Category")
        pdf.image(io.BytesIO(bar_chart_png(by_cat, "Average Reviewer Score by Category", "Score (0-3)", xlim=(0, 3))), w=content_w * 0.9)
        pdf.ln(3)

    if not by_source.empty:
        _ensure_space(pdf, 16 + chart_height_mm(content_w * 0.9, len(by_source), BAR_CHART_DIMS))
        _pdf_section_header(pdf, "Scores by Source")
        pdf.image(io.BytesIO(bar_chart_png(by_source, "Average Reviewer Score by Source", "Score (0-3)", xlim=(0, 3))), w=content_w * 0.9)
        pdf.ln(3)

    _pdf_section_header(pdf, "Bias Findings")
    if flagged_categories:
        text = "\n".join(f"- {cat} underperforms the average by {gap:.2f} points." for cat, gap in flagged_categories)
        _pdf_callout(pdf, text, kind="warning")
    else:
        _pdf_callout(pdf, "No category performed significantly worse than average.", kind="success")

    _pdf_section_header(pdf, "Conclusion & Recommendations")
    _pdf_body(pdf, conclusion_notes or "(not provided)")

    return bytes(pdf.output())


def build_evaluation_report_pdf(scored_df, by_aspect, total, answered, passed, failed, pass_rate) -> bytes:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()
    _pdf_header_band(pdf, "AI Tester - Evaluation Report")

    kpis = [
        ("Questions", str(total), PDF_PURPLE),
        ("Answered", str(answered), PDF_PURPLE),
        ("Passed", str(passed), PDF_GREEN),
        ("Failed", str(failed), PDF_RED),
        ("Pass Rate", f"{pass_rate:.0f}%", PDF_GREEN if pass_rate >= 70 else PDF_AMBER if pass_rate >= 40 else PDF_RED),
    ]
    content_w = pdf.w - pdf.l_margin - pdf.r_margin
    _pdf_kpi_row(pdf, kpis, content_w)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*PDF_MUTED)
    pdf.cell(0, 6, _pdf_safe(f"Pass threshold: Reviewer Score >= {PASS_THRESHOLD}/3"), ln=True)
    pdf.set_text_color(*PDF_INK)

    if not by_aspect.empty:
        _ensure_space(pdf, 16 + chart_height_mm(content_w, len(by_aspect), STACKED_CHART_DIMS))
        _pdf_section_header(pdf, "Pass / Fail by Aspect")
        pdf.image(io.BytesIO(pass_fail_by_aspect_png(by_aspect)), w=content_w)
        pdf.ln(3)

    _pdf_section_header(pdf, "Detailed Results")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_fill_color(255, 255, 255)
    with pdf.table(
        col_widths=(1, 3, 5, 10, 3, 2),
        text_align=("CENTER", "LEFT", "LEFT", "LEFT", "CENTER", "CENTER"),
        headings_style=FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=PDF_PURPLE),
        cell_fill_color=(245, 243, 250),
        cell_fill_mode="ROWS",
        line_height=5,
    ) as table:
        header = table.row()
        for h in ["ID", "Category", "Aspect", "Question", "Result", "Score"]:
            header.cell(_pdf_safe(h))
        for _, r in scored_df.iterrows():
            row = table.row()
            row.cell(_pdf_safe(r["ID"]))
            row.cell(_pdf_safe(r["Category"]))
            row.cell(_pdf_safe(r["Aspect"]))
            q = str(r["Question"])
            row.cell(_pdf_safe(q[:90] + ("..." if len(q) > 90 else "")))
            row.cell(_pdf_safe(r["Result"]))
            score = r["Reviewer Score"]
            row.cell(_pdf_safe("-" if score == -1 else str(int(score))))

    return bytes(pdf.output())


def build_bias_report_pdf(overall_avg, by_cat, by_source, flagged_categories) -> bytes:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()
    _pdf_header_band(pdf, "AI Tester - Bias Report")

    content_w = pdf.w - pdf.l_margin - pdf.r_margin
    kpis = [
        ("Overall Avg Score", f"{overall_avg:.2f} / 3", PDF_PURPLE),
        ("Categories Tested", str(len(by_cat)), PDF_PURPLE),
        ("Flagged Categories", str(len(flagged_categories)), PDF_RED if flagged_categories else PDF_GREEN),
    ]
    _pdf_kpi_row(pdf, kpis, content_w)

    if not by_cat.empty:
        _ensure_space(pdf, 16 + chart_height_mm(content_w * 0.9, len(by_cat), BAR_CHART_DIMS))
        _pdf_section_header(pdf, "Scores by Category")
        pdf.image(io.BytesIO(bar_chart_png(by_cat, "Average Reviewer Score by Category", "Score (0-3)", xlim=(0, 3))), w=content_w * 0.9)
        pdf.ln(3)

    if not by_source.empty:
        _ensure_space(pdf, 16 + chart_height_mm(content_w * 0.9, len(by_source), BAR_CHART_DIMS))
        _pdf_section_header(pdf, "Scores by Source")
        pdf.image(io.BytesIO(bar_chart_png(by_source, "Average Reviewer Score by Source", "Score (0-3)", xlim=(0, 3))), w=content_w * 0.9)
        pdf.ln(3)

    _pdf_section_header(pdf, "Bias Findings")
    if flagged_categories:
        text = "\n".join(f"- {cat} underperforms the average by {gap:.2f} points." for cat, gap in flagged_categories)
        _pdf_callout(pdf, text, kind="warning")
    else:
        _pdf_callout(pdf, "No category performed significantly worse than average.", kind="success")

    return bytes(pdf.output())


def extract_docx_text(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_about_doc(file) -> str:
    name = file.name.lower()
    if name.endswith(".docx"):
        return extract_docx_text(file)
    if name.endswith(".pdf"):
        return extract_pdf_text(file)
    return file.read().decode("utf-8", errors="ignore")


def load_dataset(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file), None
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file), None
    return None, file.read().decode("utf-8", errors="ignore")


def build_payload(template_str: str, question: str):
    payload = json.loads(template_str)

    def replace(obj):
        if isinstance(obj, dict):
            return {k: replace(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [replace(v) for v in obj]
        if isinstance(obj, str):
            return obj.replace("{{question}}", question)
        return obj

    return replace(payload)


def resolve_path(data, path: str):
    current = data
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


def build_headers(auth_type: str, auth_value: str, header_name: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if auth_type == "Bearer token" and auth_value:
        headers["Authorization"] = f"Bearer {auth_value}"
    elif auth_type == "API key header" and auth_value:
        headers[header_name or "x-api-key"] = auth_value
    return headers


def call_agent_api(url: str, headers: dict, payload: dict, auth_type: str, auth_value: str, plain_text: bool = False):
    params = {}
    if auth_type == "API key as query param" and auth_value:
        params["api_key"] = auth_value
    response = requests.post(url, headers=headers, json=payload, params=params, timeout=60)
    response.raise_for_status()
    if plain_text:
        return response.text
    return response.json()


ANTHROPIC_MODEL = "claude-sonnet-5"
DEEPSEEK_MODEL = "deepseek-chat"
GROQ_MODEL = "openai/gpt-oss-20b"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

# provider -> (chat-completions URL, model). All three are OpenAI-compatible.
OPENAI_COMPATIBLE_PROVIDERS = {
    "DeepSeek": ("https://api.deepseek.com/chat/completions", DEEPSEEK_MODEL),
    "Groq": ("https://api.groq.com/openai/v1/chat/completions", GROQ_MODEL),
    "OpenRouter": ("https://openrouter.ai/api/v1/chat/completions", OPENROUTER_MODEL),
}


def get_ai_api_key():
    api_key = st.session_state.get("ai_api_key_input")
    if api_key:
        return api_key
    for secret_name in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        try:
            val = st.secrets.get(secret_name)
        except Exception:
            val = None
        if val:
            return val
    return None


def get_ai_provider() -> str:
    api_key = get_ai_api_key()
    if api_key:
        if api_key.startswith("sk-ant-"):
            return "Anthropic (Claude)"
        if api_key.startswith("gsk_"):
            return "Groq"
        if api_key.startswith("sk-or-"):
            return "OpenRouter"
        return "DeepSeek"
    return "Anthropic (Claude)"


def get_anthropic_client():
    if anthropic is None or get_ai_provider() != "Anthropic (Claude)":
        return None
    api_key = get_ai_api_key()
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def ai_configured() -> bool:
    return bool(get_ai_api_key())


def call_ai_text(
    prompt: str,
    max_tokens: int = 4096,
    timeout: int = 120,
    max_retries: int = 3,
    max_wait: float = 60,
) -> str:
    provider = get_ai_provider()
    api_key = get_ai_api_key()
    if not api_key:
        raise RuntimeError(
            f"No {provider} API key configured. Paste one above, or add it to .streamlit/secrets.toml."
        )

    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        url, model = OPENAI_COMPATIBLE_PROVIDERS[provider]
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if model.startswith("openai/gpt-oss"):
            # Reasoning models spend completion tokens "thinking" before answering —
            # keep that light so the actual JSON answer isn't starved of tokens.
            body["reasoning_effort"] = "low"

        for attempt in range(max_retries + 1):
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
            if response.ok:
                content = response.json()["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise RuntimeError(
                        f"{provider} returned an empty response — the model likely spent its whole token "
                        "budget reasoning. Try fewer questions, or switch to a non-reasoning model."
                    )
                return content

            if response.status_code == 429 and attempt < max_retries:
                wait_match = re.search(r"try again in ([\d.]+)", response.text)
                wait_s = min(float(wait_match.group(1)) + 1 if wait_match else 15, max_wait)
                time.sleep(wait_s)
                continue

            detail = response.text[:500]
            if provider == "Groq" and response.status_code != 429:
                try:
                    models_resp = requests.get(
                        "https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=30,
                    )
                    if models_resp.ok:
                        available = [m["id"] for m in models_resp.json().get("data", [])]
                        detail += f"\nModels available to this key: {', '.join(available) or '(none)'}"
                except Exception:
                    pass
            raise RuntimeError(f"{provider} API error {response.status_code}: {detail}")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def dataset_summary_for_prompt(df: pd.DataFrame, text: str, max_rows: int = 15) -> str:
    if df is not None:
        cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        sample = df.head(max_rows).to_csv(index=False)
        return f"Columns: {cols}\n\nSample rows ({min(max_rows, len(df))} of {len(df)}):\n{sample}"
    if text:
        return text[:8000]
    return "(no dataset provided)"


def _extract_json_array(raw: str):
    start = raw.find("[")
    if start == -1:
        snippet = (raw or "").strip()[:400] or "(empty response)"
        raise ValueError(f"No JSON array found in the AI's response. It said: {snippet}")
    end = raw.rfind("]")
    if end != -1:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Response was likely cut off mid-array (hit the token limit) — salvage
    # whichever complete {...} objects came before the cutoff.
    decoder = json.JSONDecoder()
    rows, i, n = [], start + 1, len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n,":
            i += 1
        if i >= n or raw[i] != "{":
            break
        try:
            obj, offset = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        rows.append(obj)
        i = offset
    if not rows:
        raise ValueError("Couldn't parse any test cases from the AI's response — it may have been cut off.")
    return rows


def generate_questions_with_ai(
    dataset_summary: str,
    about_text: str,
    n_questions: int = 18,
    topic: str = "",
    aspects: list = None,
    answer_format: str = "",
    input_format: str = "",
) -> pd.DataFrame:
    aspects = aspects or ASPECT_OPTIONS
    about_text = (about_text or "")[:3000]
    dataset_summary = (dataset_summary or "")[:3000]

    topic_instruction = (
        f'\nFocus most of the questions on this specific section/topic: "{topic}". '
        "Still use the full document above for context and consistency.\n"
        if topic
        else ""
    )

    format_instruction = (
        f'\nThe AI agent under test only ever responds with: {answer_format}. Every "expected_answer" must be '
        "written as exactly one of those literal values (e.g. \"LEAVE\", \"Yes\") — never a descriptive sentence, "
        "and never prefixed with \"(inferred)\", since there is no other valid answer format to infer around.\n"
        if answer_format
        else ""
    )

    input_instruction = (
        f'\nThis AI is NOT a chatbot — it takes structured input, not natural-language questions. Every '
        f'"question" must be written as exactly this input shape, with concrete values filled in: {input_format}. '
        "Do not phrase it as a question sentence at all — write it as a structured input description.\n"
        if input_format
        else ""
    )

    prompt = f"""You are designing a black-box test suite for an AI agent, before it has been asked anything.

ABOUT THE AI (its spec / SRS / what it's supposed to do):
{about_text or "(none provided)"}

DATASET (ground truth the AI was built on):
{dataset_summary}
{topic_instruction}{input_instruction}
Write {n_questions} test cases covering these testing aspects: {", ".join(aspects)}.
Use these categories: {", ".join(CATEGORY_OPTIONS)}.
Use these sources: {", ".join(SOURCE_OPTIONS)} — "About the AI" if the question/expected answer comes from the spec above,
"Dataset" if it comes from a real row/fact in the dataset above, "General" for edge/robustness/bias probes not tied to either.

For "Expected Answer": if the spec or dataset states the correct answer directly, use that (and set Source accordingly).
If it doesn't, infer the best reasonable expected behavior yourself and prefix it with "(inferred) ".
{format_instruction}
Cover a mix of Normal, Edge Case, Tricky, and Bias Probe questions across the aspects above — don't cluster them all
in one aspect. Aim for objectively checkable questions where the dataset/spec allows it.

For each test case, also provide:
- "sub_metric": the specific dimension being tested under the aspect (e.g. "Reasoning visibility", "Demographic parity", "Boundary & edge case coverage").
- "test_objective": one sentence on what the test case is meant to confirm.
- "tools": how this would be verified in practice (e.g. "Manual review", "Paired-prompt comparison", "Regression testing").
- "covered_dimension": the underlying quality dimension this addresses.
- "lifecycle_phase": exactly one of: {", ".join(LIFECYCLE_PHASE_OPTIONS)}.

Respond with ONLY a JSON array, no prose, no markdown fences. Each element:
{{"category": "...", "aspect": "...", "source": "...", "question": "...", "expected_answer": "...", "notes": "...",
"sub_metric": "...", "test_objective": "...", "tools": "...", "covered_dimension": "...", "lifecycle_phase": "..."}}"""

    max_tokens = min(max(4096, n_questions * 350 + 1000), 16000)
    if get_ai_provider() == "Groq":
        # Groq's free on-demand tier caps prompt + completion tokens together
        # (observed limit: 8000 TPM) — leave a safety margin under that.
        prompt_tokens_est = len(prompt) // 4 + 200
        max_tokens = max(600, min(max_tokens, 7500 - prompt_tokens_est))
    raw = call_ai_text(prompt, max_tokens=max_tokens)
    rows = _extract_json_array(raw)
    if len(rows) < n_questions:
        st.warning(
            f"AI's response was cut off — got {len(rows)} of the {n_questions} requested test cases. "
            "Try a smaller number of questions, or generate again to add more."
        )

    return pd.DataFrame(
        [
            {
                "ID": str(i + 1),
                "Category": r.get("category", "Uncategorized"),
                "Aspect": r.get("aspect", "General"),
                "Sub-Metric": r.get("sub_metric", ""),
                "Source": r.get("source", "General"),
                "Question": r.get("question", ""),
                "Test Objective": r.get("test_objective", ""),
                "Expected Answer": r.get("expected_answer", ""),
                "AI Answer": "",
                "Tools": r.get("tools", ""),
                "Covered Dimension": r.get("covered_dimension", ""),
                "Lifecycle Phase": r.get("lifecycle_phase", ""),
                "Notes": r.get("notes", ""),
            }
            for i, r in enumerate(rows)
        ]
    )


def ai_judge_answer(question: str, expected: str, actual: str, about_text: str = "") -> tuple:
    prompt = f"""You are judging one answer from an AI agent under test.

Context on what this AI is supposed to do: {about_text[:2000] or "(none provided)"}

Question asked: {question}
Expected answer / what a correct answer should look like: {expected}
The AI agent's actual answer: {actual}

Score the actual answer against the expected answer on this rubric:
0 = No match / wrong / did not answer
1 = Partially matches, missing key parts or partly wrong
2 = Mostly matches, minor gaps
3 = Fully matches the expected answer or expected behavior

Respond with ONLY a JSON object, no prose: {{"score": 0-3, "reasoning": "one short sentence"}}"""

    # Judge calls are short and run once per test case in a tight loop — cap the
    # timeout/retry budget hard so a rate-limited key can't block the whole UI
    # (and the browser's connection) for minutes on a single row.
    raw = call_ai_text(prompt, max_tokens=300, timeout=30, max_retries=1, max_wait=15)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in judge response: {raw[:200]!r}")
    result = json.loads(raw[start:end + 1])
    return int(result.get("score", 0)), str(result.get("reasoning", ""))


def login_screen():
    st.markdown(
        '<div style="max-width:420px;margin:8vh auto 0 auto;text-align:center;">'
        '<h1 style="margin-bottom:4px;">AI Tester</h1>'
        '<p style="color:#6B7280;margin-top:0;">Sign in to test and score an AI agent.</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            row = get_user_by_username(username.strip())
            if row and verify_password(password, row[3], row[2]):
                st.session_state["user"] = {"id": row[0], "username": row[1], "role": row[4]}
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        st.caption("First time here? Default admin login is **admin / admin123** — change it after logging in.")


if "user" not in st.session_state:
    login_screen()
    st.stop()

CURRENT_USER = st.session_state["user"]
IS_ADMIN = CURRENT_USER["role"] == "admin"

for key, default in [
    ("df", None),
    ("dataset_df", None),
    ("dataset_text", None),
    ("dataset_name", None),
    ("about_text", None),
    ("about_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("AI Tester")
st.caption("A one-time evaluation of how well an existing AI agent answers questions.")

TAB_LABELS = ["Projects", "Setup", "Test & Score", "Results"]
if IS_ADMIN:
    TAB_LABELS = TAB_LABELS + ["Users"]

if "_goto_tab" in st.session_state:
    st.session_state["active_tab"] = st.session_state.pop("_goto_tab")

with st.sidebar:
    user_col, logout_col = st.columns([2, 1])
    with user_col:
        st.markdown(f"**{CURRENT_USER['username']}**")
        st.caption("Admin" if IS_ADMIN else "Tester")
    with logout_col:
        if st.button("Log out", key="logout_btn", use_container_width=True):
            del st.session_state["user"]
            st.rerun()
    st.markdown("### Navigation")
    active_tab = st.radio(
        "Navigation", TAB_LABELS, key="active_tab", label_visibility="collapsed"
    )

if active_tab == "Projects":
    st.markdown("### Projects")
    st.caption(
        "Every AI agent you test lives under a project — its dataset, spec, test cases, and saved run "
        "history all stay filed under it. Create one, or open one you've tested before to jump back into it."
    )

    with st.form("projects_tab_new_project_form", clear_on_submit=True):
        st.markdown("##### + Add new project")
        new_proj_col1, new_proj_col2 = st.columns(2)
        with new_proj_col1:
            new_name = st.text_input("Project name", placeholder="e.g. AI HR Tool")
        with new_proj_col2:
            new_short = st.text_input("Short name (used in file names, optional)", placeholder="e.g. HR")
        created = st.form_submit_button("Add Project", type="primary")
    if created:
        if not new_name.strip():
            st.error("Enter a project name.")
        else:
            short = new_short.strip() or re.sub(r"[^A-Za-z0-9]+", "", new_name.strip())[:12] or "Project"
            pid = create_project(new_name.strip(), short, CURRENT_USER["id"])
            st.session_state["current_project_id"] = pid
            st.session_state["current_project_name"] = new_name.strip()
            st.session_state["current_project_short"] = short
            st.session_state["_goto_tab"] = "Setup"
            st.success(f"Project '{new_name.strip()}' created — taking you to Setup.")
            st.rerun()

    st.markdown("##### Your projects" if not IS_ADMIN else "##### All projects")
    projects_df = fetch_projects(CURRENT_USER["id"], CURRENT_USER["role"])
    if projects_df.empty:
        st.info("No projects yet — add one above to get started.")
    else:
        for _, proj in projects_df.iterrows():
            proj_id = int(proj["id"])
            runs_count = len(fetch_runs(project_id=proj_id))
            is_current = st.session_state.get("current_project_id") == proj["id"]
            can_delete = IS_ADMIN or proj["user_id"] == CURRENT_USER["id"]
            p_col1, p_col2, p_col3, p_col4 = st.columns([3, 2, 1, 1])
            with p_col1:
                label = f"**{proj['name']}**" + (" 🟣 *(current)*" if is_current else "")
                st.markdown(label)
                st.caption(f"Short name: {proj['short_name'] or '—'} · Created: {proj['created_at'][:10]}")
            with p_col2:
                st.caption(f"{runs_count} run(s) saved to history")
            with p_col3:
                if st.button("Open →", key=f"open_project_{proj_id}"):
                    st.session_state["current_project_id"] = proj_id
                    st.session_state["current_project_name"] = proj["name"]
                    st.session_state["current_project_short"] = proj["short_name"]
                    st.session_state["_goto_tab"] = "Setup"
                    st.rerun()
            with p_col4:
                if can_delete:
                    if st.button("Delete", key=f"delete_project_{proj_id}"):
                        st.session_state[f"confirm_delete_{proj_id}"] = True
                        st.rerun()
            if st.session_state.get(f"confirm_delete_{proj_id}"):
                st.warning(
                    f"Delete '{proj['name']}' and all {runs_count} saved run(s)? This can't be undone."
                )
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Yes, delete it", key=f"confirm_delete_yes_{proj_id}", type="primary"):
                        delete_project(proj_id)
                        del st.session_state[f"confirm_delete_{proj_id}"]
                        if st.session_state.get("current_project_id") == proj_id:
                            for key in ("current_project_id", "current_project_name", "current_project_short"):
                                st.session_state.pop(key, None)
                        st.success(f"Deleted '{proj['name']}'.")
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key=f"confirm_delete_no_{proj_id}"):
                        del st.session_state[f"confirm_delete_{proj_id}"]
                        st.rerun()
            st.markdown("---")

if active_tab == "Setup":
    if st.session_state.get("current_project_id") is None:
        st.warning("No project selected. Head to the **Projects** tab to add or open one before continuing.")
        st.stop()

    proj_col, switch_col = st.columns([5, 1])
    with proj_col:
        st.caption(f"Working on project: **{st.session_state.get('current_project_name', '')}**")
    with switch_col:
        if st.button("Switch project"):
            st.session_state["_goto_tab"] = "Projects"
            st.rerun()

    st.markdown(
        "You give three things, then we build your test question set from them:"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="setup-card"><h4>Dataset</h4><p>The data this AI was trained on, used as ground truth to judge its answers.</p></div>', unsafe_allow_html=True)
        dataset_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "txt"], key="dataset_upl", label_visibility="collapsed")
        if dataset_file is not None and dataset_file.file_id != st.session_state.get("_dataset_file_id"):
            df_data, text_data = load_dataset(dataset_file)
            st.session_state.dataset_df = df_data
            st.session_state.dataset_text = text_data
            st.session_state.dataset_name = dataset_file.name
            st.session_state["_dataset_file_id"] = dataset_file.file_id

    with col2:
        st.markdown('<div class="setup-card"><h4>About the AI</h4><p>A document explaining what this AI is and what it\'s supposed to do.</p></div>', unsafe_allow_html=True)
        about_file = st.file_uploader("Upload About the AI document", type=["docx", "pdf", "txt"], key="about_upl", label_visibility="collapsed")
        if about_file is not None and about_file.file_id != st.session_state.get("_about_file_id"):
            st.session_state.about_text = load_about_doc(about_file)
            st.session_state.about_name = about_file.name
            st.session_state["_about_file_id"] = about_file.file_id

    with col3:
        st.markdown('<div class="setup-card"><h4>Test Cases</h4><p>Choose how to build your test question set.</p></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload test_questions.xlsx", type=["xlsx", "xls"], key="questions_upl", label_visibility="collapsed")
        if uploaded is not None and uploaded.file_id != st.session_state.get("_questions_file_id"):
            raw = pd.read_excel(uploaded)
            st.session_state.df = ensure_columns(raw)
            st.session_state["_questions_file_id"] = uploaded.file_id
            reset_question_editors()
            st.rerun()

        questions_mode = st.radio(
            "Test questions mode",
            ["Upload", "Generate automatically with AI", "Download Template"],
            key="setup_questions_mode",
            label_visibility="collapsed",
        )
        if questions_mode == "Download Template":
            st.download_button(
                "Download Test Question Template",
                data=df_to_excel_bytes(blank_template()),
                file_name="test_questions_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        elif questions_mode == "Generate automatically with AI":
            st.caption("Configure and generate below ↓")

    if questions_mode == "Generate automatically with AI":
        st.markdown("#### Generate Test Questions with AI")
        st.markdown("###### Step A — Add your AI API key")
        st.caption(
            "This powers the AI that writes your test cases. Paste an Anthropic, DeepSeek, Groq, or "
            "OpenRouter API key — kept only for this session, or set ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / "
            "GROQ_API_KEY / OPENROUTER_API_KEY in .streamlit/secrets.toml. New to this? Groq "
            "(console.groq.com/keys) or OpenRouter (openrouter.ai/keys) both have a free tier."
        )
        st.text_input(
            "AI API key",
            type="password",
            key="ai_api_key_input",
            placeholder="sk-ant-... / gsk_... / sk-or-... / sk-...",
            label_visibility="collapsed",
        )

        if not ai_configured():
            st.warning("No AI API key configured yet. Paste one above to unlock the rest of this step.")
        else:
            st.success("API key set.")
            st.markdown("###### Step B — Source materials")
            st.caption("Dataset (optional) and About the AI (required) are the same files uploaded in cards 1 & 2 above.")
            if st.session_state.about_text is None:
                st.warning("Upload an About the AI document in card 2 above first.")

            st.markdown("###### Step C — Configure & generate")
            st.caption(
                "AI reads your About-the-AI (SRS) document and drafts a full test question bank from it. "
                "If you also uploaded a dataset, AI will pull real rows as ground truth for the Expected Answer "
                "wherever it can — otherwise it infers a reasonable expected answer from the spec alone."
            )
            n_q = st.number_input(
                "How many questions?", min_value=1, value=18, step=1, key="ai_n_questions",
                help="Type any number — there's no fixed cap.",
            )
            topic = st.text_input(
                "Topic / section to focus on (optional)",
                key="ai_topic",
                placeholder='e.g. "Section 3.2: Login Flow" — leave blank to cover the whole document',
            )
            aspects = st.multiselect(
                "Testing aspects to cover",
                options=ASPECT_OPTIONS,
                default=ASPECT_OPTIONS,
                key="ai_aspects",
            )
            input_format = st.text_input(
                "How does this AI take its input? (optional, but important)",
                key="ai_input_format",
                placeholder='e.g. "Satisfaction=X%, Evaluation=Y%, Projects=N, Monthly Hours=H, Years at Company=Y, '
                            'Work Accident=Yes/No, Promoted=Yes/No, Salary=Low/Medium/High" — leave blank for a normal chat question',
                help=(
                    "If this isn't a chatbot — e.g. it predicts an outcome from structured fields like a "
                    "dataset row — describe that exact input shape here so every 'Question' is written as "
                    "one of those structured inputs instead of a natural-language question."
                ),
            )
            answer_format = st.text_input(
                "What does this AI's answer always look like? (optional, but important)",
                key="ai_answer_format",
                placeholder='e.g. "exactly LEAVE or STAY", "yes or no", "a number" — leave blank for open-ended answers',
                help=(
                    "If this AI only ever returns one of a fixed set of answers (a classifier, a "
                    "yes/no tool, etc.), tell AI here so every Expected Answer matches that exact "
                    "format — otherwise scoring will fail everything just for wording."
                ),
            )
            if st.button("Generate Test Questions with AI", type="primary"):
                if st.session_state.about_text is None:
                    st.error("Upload an About the AI document first (card 2 above).")
                elif not aspects:
                    st.error("Pick at least one testing aspect.")
                else:
                    with st.spinner("AI is drafting the test suite..."):
                        try:
                            summary = dataset_summary_for_prompt(st.session_state.dataset_df, st.session_state.dataset_text)
                            generated = generate_questions_with_ai(
                                summary, st.session_state.about_text, int(n_q), topic=topic, aspects=aspects,
                                answer_format=answer_format, input_format=input_format,
                            )
                            st.session_state.df = ensure_columns(generated)
                            st.session_state["_questions_file_id"] = "ai_generated"
                            reset_question_editors()
                            st.session_state["_last_generated_counts"] = (
                                generated["Aspect"].value_counts().to_dict()
                            )
                            st.success(f"Generated {len(generated)} test questions. Review, edit, add, or delete them below — every cell is editable.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI couldn't generate questions: {e}")

            if st.session_state.get("_last_generated_counts"):
                st.markdown("**Test cases written per section:**")
                for section, count in st.session_state["_last_generated_counts"].items():
                    st.caption(f"• {section}: {count}")

    st.caption(
        "Either way, write your own questions using the template above, or fill in the Source column with "
        "'Dataset' / 'About the AI' / 'General' to mark where each question came from — and every answer, "
        "expected or actual, stays editable by anyone afterward."
    )

    st.markdown("### Current Test Cases")
    if st.session_state.df is None:
        st.info("No test cases yet — use the '3. Test Questions' options above to upload, generate, or download a template.")
    else:
        st.caption(f"{len(st.session_state.df)} test cases. Add, edit, or delete rows directly below — right-click a row for row options.")
        edited_setup = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_qa_setup",
            column_config={
                "Category": st.column_config.SelectboxColumn(options=CATEGORY_OPTIONS),
                "Source": st.column_config.SelectboxColumn(options=SOURCE_OPTIONS),
                "Aspect": st.column_config.SelectboxColumn(options=ASPECT_OPTIONS),
                "Lifecycle Phase": st.column_config.SelectboxColumn(options=LIFECYCLE_PHASE_OPTIONS),
            },
        )
        st.session_state.df = ensure_columns(edited_setup)

    st.markdown("### Reference Materials")
    st.caption("Keep these open while you write questions — they're your source of truth.")

    ref_a, ref_b = st.columns(2)

    with ref_a:
        st.markdown("#### Dataset")
        if st.session_state.dataset_df is not None:
            data = st.session_state.dataset_df
            st.markdown(f"**{st.session_state.dataset_name}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", len(data))
            m2.metric("Columns", len(data.columns))
            m3.metric("Missing values", int(data.isna().sum().sum()))
            st.dataframe(data.head(50), use_container_width=True, height=300)
        elif st.session_state.dataset_text is not None:
            st.markdown(f"**{st.session_state.dataset_name}**")
            st.text_area("Dataset preview", st.session_state.dataset_text, height=300, disabled=True, label_visibility="collapsed")
        else:
            st.info("No dataset uploaded yet.")

    with ref_b:
        st.markdown("#### About the AI")
        if st.session_state.about_text is not None:
            st.markdown(f"**{st.session_state.about_name}**")
            st.text_area("About the AI text", st.session_state.about_text, height=350, disabled=True, label_visibility="collapsed")
        else:
            st.info("No document uploaded yet.")

if active_tab == "Test & Score":
    if st.session_state.df is None:
        st.info("Upload `test_questions.xlsx` in the Setup tab first.")
    else:
        st.subheader("Send Questions to the AI")
        st.markdown(
            "Connect to the AI agent below to send every question automatically, or just paste answers "
            "into the **AI Answer** column by hand — either works, and you can mix both."
        )

        agent_mode = st.radio(
            "How is the AI agent reachable?",
            ["API (HTTP)", "No API — I'll paste answers manually"],
            key="agent_mode",
            horizontal=True,
        )

        if agent_mode == "No API — I'll paste answers manually":
            st.info(
                "No problem — send each question to the AI agent yourself (chat UI, app, whatever it has), "
                "then type or paste its answer directly into the **AI Answer** column in the table below."
            )

        if agent_mode == "API (HTTP)":
            with st.expander("Connect the AI agent's API", expanded=True):
                api_url = st.text_input("API endpoint URL", key="api_url", placeholder="https://example.com/api/chat")

                col_a, col_b = st.columns(2)
                with col_a:
                    auth_type = st.selectbox(
                        "Authentication",
                        ["None", "Bearer token", "API key header", "API key as query param"],
                        key="api_auth_type",
                    )
                with col_b:
                    auth_value = st.text_input("API key / token", key="api_auth_value", type="password")

                header_name = "x-api-key"
                if auth_type == "API key header":
                    header_name = st.text_input("Header name", value="x-api-key", key="api_header_name")

                body_template = st.text_area(
                    "Request body template — {{question}} gets replaced with each question",
                    value='{"question": "{{question}}"}',
                    key="api_body_template",
                    height=80,
                )
                plain_text_response = st.checkbox(
                    "Response is plain text (not JSON) — e.g. a streamed chat reply",
                    value=False,
                    key="api_plain_text",
                )
                response_path = st.text_input(
                    "Where's the answer in the response? (dot path, e.g. answer or data.reply)",
                    value="answer",
                    key="api_response_path",
                    disabled=plain_text_response,
                )

                test_col, send_col = st.columns(2)

                with test_col:
                    if st.button("Test connection"):
                        if not api_url:
                            st.error("Enter an API endpoint URL first.")
                        else:
                            try:
                                headers = build_headers(auth_type, auth_value, header_name)
                                payload = build_payload(body_template, "Hello, what can you help me with?")
                                with st.expander("Request sent (debug)"):
                                    st.write("Headers:", {k: v for k, v in headers.items()})
                                    st.write("Payload:", payload)
                                raw = call_agent_api(api_url, headers, payload, auth_type, auth_value, plain_text_response)
                                answer = raw.strip() if plain_text_response else resolve_path(raw, response_path)
                                st.success("Connected. Here's what came back:")
                                if plain_text_response:
                                    st.text(raw)
                                else:
                                    st.json(raw)
                                if answer:
                                    st.markdown(f"**Extracted answer:** {answer}")
                                else:
                                    st.warning(
                                        "Got a response but couldn't find an answer at that path — adjust "
                                        "'Where's the answer in the response?' to match the JSON above."
                                    )
                            except json.JSONDecodeError:
                                st.error("Request body template isn't valid JSON.")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Request failed: {e}")

                with send_col:
                    overwrite = st.checkbox("Overwrite existing AI Answers", value=False, key="api_overwrite")
                    if st.button("Send all questions to the AI", type="primary"):
                        if not api_url:
                            st.error("Enter an API endpoint URL first.")
                        else:
                            work = st.session_state.df.copy()
                            headers = build_headers(auth_type, auth_value, header_name)
                            progress = st.progress(0.0)
                            status = st.empty()
                            total = len(work)
                            for i, (idx, row) in enumerate(work.iterrows()):
                                already_answered = str(row["AI Answer"]).strip() != ""
                                if already_answered and not overwrite:
                                    progress.progress((i + 1) / total)
                                    continue
                                status.text(f"Sending question {i + 1} of {total}...")
                                try:
                                    payload = build_payload(body_template, str(row["Question"]))
                                    raw = call_agent_api(api_url, headers, payload, auth_type, auth_value, plain_text_response)
                                    answer = raw.strip() if plain_text_response else resolve_path(raw, response_path)
                                    if answer:
                                        work.at[idx, "AI Answer"] = str(answer)
                                    else:
                                        work.at[idx, "Notes"] = "ERROR: no answer found at response path"
                                except json.JSONDecodeError:
                                    work.at[idx, "Notes"] = "ERROR: request body template isn't valid JSON"
                                    break
                                except requests.exceptions.RequestException as e:
                                    work.at[idx, "Notes"] = f"ERROR: {e}"
                                progress.progress((i + 1) / total)
                                time.sleep(0.3)
                            status.text("Done.")
                            st.session_state.df = ensure_columns(work)
                            reset_question_editors()
                            st.rerun()

        edited = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_qa",
            column_config={
                "Category": st.column_config.SelectboxColumn(options=CATEGORY_OPTIONS),
                "Source": st.column_config.SelectboxColumn(options=SOURCE_OPTIONS),
                "Aspect": st.column_config.SelectboxColumn(options=ASPECT_OPTIONS),
                "Lifecycle Phase": st.column_config.SelectboxColumn(options=LIFECYCLE_PHASE_OPTIONS),
            },
        )
        st.session_state.df = ensure_columns(edited)

    st.markdown("---")
    if st.session_state.df is None:
        st.info("Upload `test_questions.xlsx` in the Setup tab first.")
    else:
        st.subheader("Evaluate the Answers")
        st.markdown(
            "Now we look at the answers and score them. For each answer we ask:\n\n"
            "- Was it correct? (**Functional Correctness**)\n"
            "- Was it accurate and useful? (**Model Accuracy**)\n"
            "- Was it fair? (**Bias & Fairness** — measured in the next tab, across all questions)\n"
            "- Did it stay consistent? (**Reliability** — ask the same question twice and compare)\n"
            "- Could we understand how it got there? (**Explainability** — use the Notes column)"
        )
        st.caption(
            "The button below automatically scores Functional Correctness / Accuracy by comparing text similarity "
            "to your Expected Answer. It can't judge fairness, consistency, or explainability by itself — those need "
            "your judgment, using the Reviewer Score and Notes columns, and the dedicated Bias tab. "
            f"A question Passes once its Reviewer Score is {PASS_THRESHOLD} or higher (out of 3)."
        )

        st.markdown("##### Choose a scoring method")
        scoring_options = [
            "Text similarity (free, offline)",
            "AI judge (uses API, reads meaning not just wording)",
        ]
        scoring_method = st.radio("Scoring method", scoring_options, key="scoring_method", horizontal=True)

        if scoring_method.startswith("AI judge"):
            st.caption(
                "Paste an Anthropic, DeepSeek, Groq, or OpenRouter API key so AI can compare each Expected "
                "Answer against the actual answer by meaning, not just wording."
            )
            st.text_input(
                "AI API key",
                type="password",
                key="ai_api_key_input",
                placeholder="sk-ant-... / gsk_... / sk-or-... / sk-...",
                label_visibility="collapsed",
            )
            if ai_configured():
                st.success("AI API key set.")
            else:
                st.warning("Paste an API key above to use AI judge scoring.")

        if st.button("Run automatic scoring", type="primary"):
          use_ai_judge = scoring_method.startswith("AI judge")
          if use_ai_judge and not ai_configured():
            st.error("Paste an API key above first to use AI judge scoring.")
          else:
            work = st.session_state.df.copy()
            ratios, auto_scores, reasonings = [], [], []
            progress = st.progress(0.0) if use_ai_judge else None
            status_text = st.empty() if use_ai_judge else None
            if use_ai_judge:
                st.caption(
                    "AI judge calls the API once per test case — this can take a little while. "
                    "Keep this tab open; if your connection is slow, Text similarity scoring is instant."
                )
            for i, (_, row) in enumerate(work.iterrows()):
                answered = str(row["AI Answer"]).strip() != ""
                if not answered:
                    ratios.append(0.0)
                    auto_scores.append(to_rubric_score(0.0, False))
                    reasonings.append("")
                elif use_ai_judge:
                    if status_text is not None:
                        status_text.caption(f"Scoring {i + 1}/{len(work)} with AI judge…")
                    try:
                        score, reasoning = ai_judge_answer(
                            row["Question"], row["Expected Answer"], row["AI Answer"],
                            st.session_state.about_text or "",
                        )
                        auto_scores.append(score)
                        ratios.append(round(score / 3 * 100, 1))
                        reasonings.append(reasoning)
                    except Exception as e:
                        auto_scores.append(-1)
                        ratios.append(0.0)
                        reasonings.append(f"ERROR: {e}")
                else:
                    r = similarity_score(row["Expected Answer"], row["AI Answer"])
                    ratios.append(round(r * 100, 1))
                    auto_scores.append(to_rubric_score(r, True))
                    reasonings.append("")
                if progress is not None:
                    progress.progress((i + 1) / len(work))
            if status_text is not None:
                status_text.empty()
            work["Similarity %"] = ratios
            work["Auto Score (0-3)"] = auto_scores
            if use_ai_judge:
                work["Judge Reasoning"] = reasonings
            if "Reviewer Score" not in work.columns:
                work["Reviewer Score"] = work["Auto Score (0-3)"]
            work["Result"] = work["Reviewer Score"].apply(result_from_score)
            st.session_state.scored_df = work
            if "editor_scores" in st.session_state:
                del st.session_state["editor_scores"]
            st.toast("Scoring complete!", icon="✅")
            st.success(f"Scoring complete — {len(work)} test cases scored. Results are shown below.")

        if "scored_df" in st.session_state:
            scored = st.session_state.scored_df.copy()
            scored["Result"] = scored["Reviewer Score"].apply(result_from_score)
            answered_mask = scored["Auto Score (0-3)"] != -1
            passed = int((scored["Result"] == "Pass").sum())
            failed = int((scored["Result"] == "Fail").sum())
            pass_rate = passed / (passed + failed) * 100 if (passed + failed) > 0 else 0

            eval_by_aspect = scored.groupby("Aspect")["Result"].value_counts().unstack(fill_value=0)
            eval_by_aspect = eval_by_aspect.reindex(columns=["Pass", "Fail", "Not Answered"], fill_value=0)
            eval_pdf_bytes = build_evaluation_report_pdf(
                scored, eval_by_aspect, len(scored), int(answered_mask.sum()), passed, failed, pass_rate
            )
            st.download_button(
                "⬇ Download Report",
                data=eval_pdf_bytes,
                file_name="evaluation_report.pdf",
                mime="application/pdf",
                type="primary",
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total questions", len(scored))
            c2.metric("Answered", int(answered_mask.sum()))
            c3.metric("Passed", passed)
            c4.metric("Failed", failed)
            c5.metric("Pass rate", f"{pass_rate:.0f}%")

            st.markdown(
                "**Review and correct scores below** (0 = No match, 3 = Full match). Result updates automatically "
                "from Reviewer Score. Expected Answer and AI Answer are editable here too — fix them and re-run "
                "scoring if needed."
            )
            reviewed = st.data_editor(
                scored,
                use_container_width=True,
                key="editor_scores",
                column_config={
                    "Reviewer Score": st.column_config.SelectboxColumn(options=[-1, 0, 1, 2, 3]),
                },
                disabled=[
                    "ID", "Category", "Aspect", "Sub-Metric", "Source", "Question", "Test Objective",
                    "Tools", "Covered Dimension", "Lifecycle Phase", "Similarity %", "Auto Score (0-3)", "Result",
                ],
            )
            reviewed["Result"] = reviewed["Reviewer Score"].apply(result_from_score)
            st.session_state.scored_df = reviewed

            st.download_button(
                "Download scored spreadsheet",
                data=df_to_excel_bytes(reviewed),
                file_name="all_answers_scored.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Click 'Run automatic scoring' to grade the answers you've collected so far.")

if active_tab == "Results":
    st.subheader("Detect Bias")
    st.markdown(
        "Bias means the AI performs noticeably worse for certain topics, question types, or phrasings than others. "
        "We check by comparing average scores across your question categories and sources — a category that scores "
        "much lower than the rest is a bias signal worth investigating."
    )
    if "scored_df" not in st.session_state:
        st.info("Run scoring in the Test & Score tab first.")
    else:
        scored = st.session_state.scored_df
        valid = scored[scored["Reviewer Score"] >= 0]
        if valid.empty:
            st.warning("No scored, answered questions yet.")
        else:
            overall_avg = valid["Reviewer Score"].mean()
            by_cat = valid.groupby("Category")["Reviewer Score"].mean().sort_values(ascending=False)
            by_source = valid.groupby("Source")["Reviewer Score"].mean().reindex(SOURCE_OPTIONS).dropna()
            gap_threshold = 0.75
            flagged = by_cat[by_cat <= overall_avg - gap_threshold]
            flagged_categories = [(cat, overall_avg - score) for cat, score in flagged.items()]

            bias_pdf_bytes = build_bias_report_pdf(overall_avg, by_cat, by_source, flagged_categories)
            st.download_button(
                "⬇ Download Report",
                data=bias_pdf_bytes,
                file_name="bias_report.pdf",
                mime="application/pdf",
                type="primary",
            )

            st.markdown("### By category")
            st.bar_chart(by_cat, color="#4FA8D8")

            st.markdown(f"**Overall average score:** {overall_avg:.2f} / 3")
            if flagged_categories:
                st.markdown("### Categories performing notably worse")
                for cat, gap in flagged_categories:
                    st.markdown(
                        f'<div class="flag-box"><b>{cat}</b>: {overall_avg - gap:.2f} / 3 '
                        f"— {gap:.2f} points below the overall average. "
                        "Possible bias or a gap in this topic.</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No category is significantly below the overall average.")

            st.markdown("### By source (Dataset vs About the AI vs General)")
            st.bar_chart(by_source, color="#4FA8D8")
            st.caption(
                "A low 'About the AI' score means the agent isn't doing what it's described to do. "
                "A low Dataset score means its answers don't match what it was trained on."
            )

    st.markdown("---")
    st.subheader("Conclusion & Analysis")
    st.markdown("This is the deliverable — everything else was in service of this.")
    if "scored_df" not in st.session_state:
        st.info("Run scoring in the Test & Score tab first.")
    else:
        scored = st.session_state.scored_df.copy()
        scored["Result"] = scored["Reviewer Score"].apply(result_from_score)
        valid = scored[scored["Reviewer Score"] >= 0]

        agent_desc = st.session_state.get("agent_desc_input", "")

        overall_avg = valid["Reviewer Score"].mean() if not valid.empty else 0
        by_cat = valid.groupby("Category")["Reviewer Score"].mean().sort_values(ascending=False) if not valid.empty else pd.Series(dtype=float)
        by_source = valid.groupby("Source")["Reviewer Score"].mean() if not valid.empty else pd.Series(dtype=float)
        passed = int((scored["Result"] == "Pass").sum())
        failed = int((scored["Result"] == "Fail").sum())
        not_answered = int((scored["Result"] == "Not Answered").sum())
        pass_rate = passed / (passed + failed) * 100 if (passed + failed) > 0 else 0
        answered_count = int((scored["AI Answer"].astype(str).str.strip() != "").sum())

        if not scored.empty:
            by_aspect = scored.groupby("Aspect")["Result"].value_counts().unstack(fill_value=0)
            by_aspect = by_aspect.reindex(columns=["Pass", "Fail", "Not Answered"], fill_value=0)
        else:
            by_aspect = pd.DataFrame(columns=["Pass", "Fail", "Not Answered"])

        flagged_categories = []
        if not by_cat.empty:
            flagged = by_cat[by_cat <= overall_avg - 0.75]
            flagged_categories = [(cat, overall_avg - score) for cat, score in flagged.items()]

        auto_conclusion = generate_conclusion_text(
            passed, failed, not_answered, pass_rate, overall_avg, flagged_categories, by_aspect
        )
        if "conclusion_notes_input" not in st.session_state:
            st.session_state["conclusion_notes_input"] = auto_conclusion
        conclusion_notes = st.session_state.get("conclusion_notes_input", "")

        pdf_bytes = build_pdf_report(
            agent_desc,
            st.session_state.dataset_name,
            st.session_state.about_name,
            len(scored),
            answered_count,
            passed,
            failed,
            not_answered,
            pass_rate,
            overall_avg,
            by_aspect,
            by_cat,
            by_source,
            flagged_categories,
            conclusion_notes,
        )

        dl_col, save_col = st.columns(2)
        with dl_col:
            st.download_button(
                "⬇ Download Report",
                data=pdf_bytes,
                file_name="assessment_report.pdf",
                mime="application/pdf",
                type="primary",
            )
        with save_col:
            if st.button("💾 Save Run to History"):
                if st.session_state.get("current_project_id") is None:
                    st.error("Add or select a project in the Setup tab first.")
                else:
                    run_id = save_run(
                        st.session_state["current_project_id"],
                        st.session_state.get("current_project_name", ""),
                        agent_desc,
                        st.session_state.dataset_name,
                        st.session_state.about_name,
                        len(scored),
                        answered_count,
                        passed,
                        failed,
                        not_answered,
                        pass_rate,
                        overall_avg,
                        scored,
                    )
                    st.success(f"Saved as run #{run_id}. View trends in the Dashboard tab.")
        st.caption("Updates automatically as you edit the fields below — just click Download again when ready.")

        project_name_display = st.session_state.get("current_project_name", "")
        st.caption(
            f"Testing project: **{project_name_display}**" if project_name_display
            else "Tip: add or select a project in the **Setup** tab to label this report and its saved history."
        )

        st.markdown("##### What is this AI agent supposed to do?")
        st.text_area("What is this AI agent supposed to do?", height=80, key="agent_desc_input", label_visibility="collapsed")

        concl_label_col, concl_btn_col = st.columns([5, 1])
        with concl_label_col:
            st.markdown("##### Your conclusion / recommendations")
            st.caption("Written automatically from your results — edit it if you want, or regenerate it after rescoring.")
        with concl_btn_col:
            if st.button("🔄 Regenerate"):
                st.session_state["conclusion_notes_input"] = auto_conclusion
                st.rerun()
        st.text_area(
            "Your conclusion / recommendations (what needs to improve, is it safe to use as-is?)",
            height=100,
            key="conclusion_notes_input",
            label_visibility="collapsed",
        )

        with st.expander("Plain-text version (for copy/paste or quick review)"):
            lines = [
                "AI TESTER — CONCLUSION & ANALYSIS REPORT",
                f"Generated: {date.today().isoformat()}",
                "",
                "WHAT WE TESTED",
                agent_desc or "(not provided)",
                "",
                "REFERENCE MATERIALS USED",
                f"  Dataset: {st.session_state.dataset_name or '(none provided)'}",
                f"  About the AI document: {st.session_state.about_name or '(none provided)'}",
                "",
                "SUMMARY",
                f"Questions tested: {len(scored)}",
                f"Answered: {answered_count}",
                f"Passed: {passed}  Failed: {failed}  Not answered: {not_answered}  (pass threshold: Reviewer Score >= {PASS_THRESHOLD}/3)",
                f"Pass rate: {pass_rate:.0f}%",
                f"Overall average score: {overall_avg:.2f} / 3",
                "",
                "PASS/FAIL BY ASPECT",
            ]
            if not by_aspect.empty:
                for aspect in by_aspect.index:
                    p, f, na = (int(by_aspect.loc[aspect, c]) for c in ("Pass", "Fail", "Not Answered"))
                    rate = (p / (p + f) * 100) if (p + f) > 0 else 0
                    lines.append(f"  - {aspect}: {p} passed / {f} failed / {na} not answered ({rate:.0f}% pass rate)")
            else:
                lines.append("  (no scored questions yet)")
            lines += ["", "SCORES BY CATEGORY"]
            for cat, score in by_cat.items():
                lines.append(f"  - {cat}: {score:.2f} / 3")
            lines += ["", "SCORES BY SOURCE"]
            for src, score in by_source.items():
                lines.append(f"  - {src}: {score:.2f} / 3")
            lines += ["", "BIAS FINDINGS"]
            if flagged_categories:
                for cat, gap in flagged_categories:
                    lines.append(f"  - {cat} underperforms the average by {gap:.2f} points.")
            else:
                lines.append("  - No category performed significantly worse than average.")
            lines += ["", "CONCLUSION & RECOMMENDATIONS", conclusion_notes or "(not provided)"]

            report_text = "\n".join(lines)
            st.text_area("Report preview", report_text, height=300, label_visibility="collapsed")
            st.download_button(
                "Download plain text (.txt)",
                data=report_text.encode("utf-8"),
                file_name="assessment_report.txt",
                mime="text/plain",
            )

        st.markdown("### Final Test Results Report")
        st.caption(
            "One row per test case in a standard reporting format (ID, Aspect, Category, Source, Test Input, "
            "What to Check / Expected, Actual Result, Pass/Fail, Notes) — the final deliverable spreadsheet."
        )
        results_report = build_final_results_report(scored)

        result_filter = st.multiselect(
            "Include which results? (leave empty for all)",
            options=["Pass", "Fail", "Not Answered"],
            key="final_report_result_filter",
        )
        filtered_report = (
            results_report[results_report["Pass/Fail"].isin(result_filter)] if result_filter else results_report
        )
        st.dataframe(filtered_report, use_container_width=True, height=350)
        if result_filter:
            st.caption(f"Showing {len(filtered_report)} of {len(results_report)} test cases.")

        project_name = st.session_state.get("current_project_name", "").strip()
        project_short = st.session_state.get("current_project_short", "").strip()
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", project_short or project_name).strip("_") or "AI_Tool"
        report_summary = {
            "Project / AI tool": project_name or "(not set)",
            "What this AI is supposed to do": agent_desc or "(not provided)",
            "Generated": date.today().isoformat(),
            "Dataset used": st.session_state.dataset_name or "(none provided)",
            "About the AI document used": st.session_state.about_name or "(none provided)",
            "Total test cases": len(results_report),
            "Answered": answered_count,
            "Passed": passed,
            "Failed": failed,
            "Not answered": not_answered,
            "Pass rate": f"{pass_rate:.0f}%",
            "Overall average score (out of 3)": f"{overall_avg:.2f}",
            "This file contains": (
                f"Only: {', '.join(result_filter)} ({len(filtered_report)} of {len(results_report)} test cases)"
                if result_filter
                else f"All {len(results_report)} test cases"
            ),
        }
        file_suffix = "_".join(r.replace(" ", "") for r in result_filter) if result_filter else "RESULTS"
        st.download_button(
            "⬇ Extract Report (Excel)",
            data=report_workbook_bytes(filtered_report, "Test Cases", report_summary),
            file_name=f"{safe_name}_Test_Cases_{file_suffix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        if not project_name:
            st.caption("Tip: add or select a project in the **Setup** tab to name this file after the project you're testing.")

    st.markdown("---")
    st.subheader("Dashboard")
    st.markdown("Slice and drill into your current results, and track pass rate across saved runs over time.")

    if "scored_df" not in st.session_state:
        st.info("Run scoring in the Test & Score tab first.")
    else:
        scored = st.session_state.scored_df.copy()
        scored["Result"] = scored["Reviewer Score"].apply(result_from_score)

        st.markdown("### Slice the current run")
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            aspect_filter = st.multiselect("Aspect", sorted(scored["Aspect"].unique()))
        with f2:
            category_filter = st.multiselect("Category", sorted(scored["Category"].unique()))
        with f3:
            source_filter = st.multiselect("Source", sorted(scored["Source"].unique()))
        with f4:
            result_filter = st.multiselect("Result", ["Pass", "Fail", "Not Answered"])
        with f5:
            lifecycle_options = sorted(p for p in scored["Lifecycle Phase"].unique() if p)
            lifecycle_filter = st.multiselect("Lifecycle Phase", lifecycle_options)

        filtered = scored.copy()
        if aspect_filter:
            filtered = filtered[filtered["Aspect"].isin(aspect_filter)]
        if category_filter:
            filtered = filtered[filtered["Category"].isin(category_filter)]
        if source_filter:
            filtered = filtered[filtered["Source"].isin(source_filter)]
        if result_filter:
            filtered = filtered[filtered["Result"].isin(result_filter)]
        if lifecycle_filter:
            filtered = filtered[filtered["Lifecycle Phase"].isin(lifecycle_filter)]

        f_passed = int((filtered["Result"] == "Pass").sum())
        f_failed = int((filtered["Result"] == "Fail").sum())
        f_rate = f_passed / (f_passed + f_failed) * 100 if (f_passed + f_failed) > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matching questions", len(filtered))
        m2.metric("Passed", f_passed)
        m3.metric("Failed", f_failed)
        m4.metric("Pass rate", f"{f_rate:.0f}%")

        st.markdown("### Drill-down: matching questions")
        st.dataframe(
            filtered[["ID", "Category", "Aspect", "Lifecycle Phase", "Source", "Question", "AI Answer", "Result", "Reviewer Score"]],
            use_container_width=True,
        )

        if not filtered.empty:
            dd1, dd2 = st.columns(2)
            with dd1:
                st.markdown("**By Aspect**")
                st.bar_chart(filtered["Aspect"].value_counts(), color="#4FA8D8")
            with dd2:
                st.markdown("**By Category**")
                st.bar_chart(filtered["Category"].value_counts(), color="#4FA8D8")

    st.markdown("---")
    st.markdown("### Run history")
    all_projects_df = fetch_projects(CURRENT_USER["id"], CURRENT_USER["role"])
    project_filter_options = ["All projects"] + [
        f"{r['name']} ({r['short_name']})" if r["short_name"] else r["name"]
        for _, r in all_projects_df.iterrows()
    ]
    project_filter_choice = st.selectbox("Filter by project", project_filter_options, key="history_project_filter")
    if project_filter_choice == "All projects":
        runs_df = fetch_runs(owner_user_id=None if IS_ADMIN else CURRENT_USER["id"])
    else:
        picked_row = all_projects_df.iloc[project_filter_options.index(project_filter_choice) - 1]
        runs_df = fetch_runs(project_id=int(picked_row["id"]))

    if runs_df.empty:
        st.info("No runs saved yet. Use 'Save Run to History' in the Results tab to start tracking trends here.")
    else:
        st.dataframe(runs_df, use_container_width=True)
        if len(runs_df) > 1:
            trend = runs_df.sort_values("Saved At")[["Saved At", "Pass Rate %"]].set_index("Saved At")
            st.markdown("**Pass rate over time**")
            st.line_chart(trend, color="#4FA8D8")
        else:
            st.caption("Save at least 2 runs to see a trend line here.")

if IS_ADMIN and active_tab == "Users":
    st.subheader("Users")
    st.caption("Add teammates as testers, or promote them to admin. Testers only see their own projects; admins see everyone's.")

    with st.form("add_user_form", clear_on_submit=True):
        st.markdown("##### + Add user")
        u_col1, u_col2, u_col3 = st.columns([2, 2, 1])
        with u_col1:
            new_username = st.text_input("Username")
        with u_col2:
            new_password = st.text_input("Temporary password", type="password")
        with u_col3:
            new_role = st.selectbox("Role", ["tester", "admin"])
        add_user_submitted = st.form_submit_button("Add User", type="primary")
    if add_user_submitted:
        if not new_username.strip() or not new_password:
            st.error("Enter both a username and a password.")
        elif get_user_by_username(new_username.strip()):
            st.error(f"Username '{new_username.strip()}' is already taken.")
        else:
            create_user(new_username.strip(), new_password, new_role)
            st.success(f"User '{new_username.strip()}' added as {new_role}.")
            st.rerun()

    st.markdown("##### All users")
    users_df = fetch_users()
    st.dataframe(users_df, use_container_width=True, hide_index=True)

    st.markdown("##### Edit a user")
    st.caption("Rename a user or reset their password — including your own account (e.g. renaming 'admin').")
    edit_username = st.selectbox("Choose a user", users_df["username"], key="edit_user_select")
    edit_row = users_df[users_df["username"] == edit_username].iloc[0]
    with st.form("edit_user_form"):
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            edited_username = st.text_input("Username", value=edit_row["username"])
        with e_col2:
            edited_password = st.text_input("New password (leave blank to keep current)", type="password")
        save_edit = st.form_submit_button("Save changes", type="primary")
    if save_edit:
        edited_username = edited_username.strip()
        if not edited_username:
            st.error("Username can't be empty.")
        elif edited_username != edit_row["username"] and get_user_by_username(edited_username):
            st.error(f"Username '{edited_username}' is already taken.")
        else:
            if edited_username != edit_row["username"]:
                rename_user(int(edit_row["id"]), edited_username)
                if int(edit_row["id"]) == CURRENT_USER["id"]:
                    st.session_state["user"]["username"] = edited_username
            if edited_password:
                change_user_password(int(edit_row["id"]), edited_password)
            st.success(f"Updated '{edit_row['username']}'.")
            st.rerun()
