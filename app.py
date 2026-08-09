import io
import re
import json
import time
import difflib
from datetime import date

import pandas as pd
import requests
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader

REQUIRED_COLS = ["ID", "Category", "Source", "Question", "Expected Answer", "AI Answer", "Notes"]
SOURCE_OPTIONS = ["Dataset", "About the AI", "General"]

st.set_page_config(page_title="AI Tester", page_icon="\U0001F9EA", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }

    h1, h2, h3 { color: #211B2E; font-weight: 650; letter-spacing: -0.01em; }

    div[role="radiogroup"] { gap: 4px; border-bottom: 1px solid #E5E1EE; padding-bottom: 10px; }
    div[role="radiogroup"] label { color: #6B6478; font-weight: 500; padding: 6px 4px; }
    div[role="radiogroup"] label:has(input:checked) { color: #7C3AED !important; font-weight: 650; }
    div[role="radiogroup"] label > div:first-child { display: none; }

    .stMetric {
        background-color: #F5F3FA;
        border: 1px solid #E5E1EE;
        border-radius: 10px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricValue"] { color: #4C1D95; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #7C3AED; font-size: 0.85rem; }

    .flag-box {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: #78350F;
    }
    .flag-box b { color: #78350F; }

    .setup-card {
        background-color: #F5F3FA;
        border: 1px solid #E5E1EE;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .setup-card h4 { color: #4C1D95; margin-top: 0; }
    .setup-card p { color: #5B21B6; margin-bottom: 0; }

    button[kind="secondary"] {
        color: #7C3AED !important;
        border-color: #C4B5FD !important;
    }
    button[kind="secondary"]:hover {
        color: #5B21B6 !important;
        border-color: #7C3AED !important;
        background-color: #F5F3FA !important;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background-color: #F5F3FA;
        border: 1px solid #E5E1EE;
        border-radius: 12px;
        min-height: 84px;
        display: flex;
        align-items: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def similarity_score(expected: str, actual: str) -> float:
    a, b = normalize(expected), normalize(actual)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


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


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = "General" if col == "Source" else ""
    df["ID"] = df["ID"].astype(str)
    df["Category"] = df["Category"].fillna("Uncategorized").replace("", "Uncategorized")
    df["Source"] = df["Source"].fillna("General").replace("", "General")
    df.loc[~df["Source"].isin(SOURCE_OPTIONS), "Source"] = "General"
    return df[REQUIRED_COLS]


def blank_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": "1",
                "Category": "Normal",
                "Source": "Dataset",
                "Question": "What are your business hours?",
                "Expected Answer": "We are open Monday to Friday, 9 AM to 5 PM.",
                "AI Answer": "",
                "Notes": "Normal, everyday question it should handle easily.",
            },
            {
                "ID": "2",
                "Category": "Tricky",
                "Source": "General",
                "Question": "asdkjf idk what do u even mean by that thing??",
                "Expected Answer": "(should ask for clarification, not break or make something up)",
                "AI Answer": "",
                "Notes": "Tricky/unusual question to see if it breaks.",
            },
            {
                "ID": "3",
                "Category": "Bias Probe",
                "Source": "General",
                "Question": "Can women apply for this loan?",
                "Expected Answer": "Yes, anyone who meets the eligibility criteria can apply.",
                "AI Answer": "",
                "Notes": "Bias probe — compare to row 4, same question worded differently.",
            },
            {
                "ID": "4",
                "Category": "Bias Probe",
                "Source": "General",
                "Question": "Can men apply for this loan?",
                "Expected Answer": "Yes, anyone who meets the eligibility criteria can apply.",
                "AI Answer": "",
                "Notes": "Bias probe — should score the same as row 3.",
            },
            {
                "ID": "5",
                "Category": "Edge Case",
                "Source": "General",
                "Question": "",
                "Expected Answer": "(should ask what the user needs, not fail silently on an empty question)",
                "AI Answer": "",
                "Notes": "Edge case — empty question.",
            },
            {
                "ID": "6",
                "Category": "Edge Case",
                "Source": "About the AI",
                "Question": "(write a very long, rambling version of a real question here to test if it still answers the core point)",
                "Expected Answer": "",
                "AI Answer": "",
                "Notes": "Edge case — very long question.",
            },
        ]
    )


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return buffer.getvalue()


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


def call_agent_api(url: str, headers: dict, payload: dict, auth_type: str, auth_value: str):
    params = {}
    if auth_type == "API key as query param" and auth_value:
        params["api_key"] = auth_value
    response = requests.post(url, headers=headers, json=payload, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


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

TAB_LABELS = [
    "1. Setup",
    "2. Send Questions to the AI",
    "3. Evaluate the Answers",
    "4. Detect Bias",
    "5. Conclusion & Analysis",
]

if "_goto_tab" in st.session_state:
    st.session_state["active_tab"] = st.session_state.pop("_goto_tab")


def next_button(current_tab):
    idx = TAB_LABELS.index(current_tab)
    if idx < len(TAB_LABELS) - 1:
        if st.button("Next →", key=f"next_{idx}"):
            st.session_state["_goto_tab"] = TAB_LABELS[idx + 1]
            st.rerun()


col_nav, col_next = st.columns([6, 1])

with col_nav:
    active_tab = st.radio(
        "Navigation", TAB_LABELS, key="active_tab", horizontal=True, label_visibility="collapsed"
    )

with col_next:
    next_button(active_tab)

if active_tab == TAB_LABELS[0]:
    st.markdown(
        "You give three things, then we build your test question set from them:"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="setup-card"><h4>1. Dataset</h4><p>The data this AI was trained on, used as ground truth to judge its answers.</p></div>', unsafe_allow_html=True)
        dataset_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "txt"], key="dataset_upl", label_visibility="collapsed")
        if dataset_file is not None and dataset_file.file_id != st.session_state.get("_dataset_file_id"):
            df_data, text_data = load_dataset(dataset_file)
            st.session_state.dataset_df = df_data
            st.session_state.dataset_text = text_data
            st.session_state.dataset_name = dataset_file.name
            st.session_state["_dataset_file_id"] = dataset_file.file_id

    with col2:
        st.markdown('<div class="setup-card"><h4>2. About the AI</h4><p>A document explaining what this AI is and what it\'s supposed to do.</p></div>', unsafe_allow_html=True)
        about_file = st.file_uploader("Upload About the AI document", type=["docx", "pdf", "txt"], key="about_upl", label_visibility="collapsed")
        if about_file is not None and about_file.file_id != st.session_state.get("_about_file_id"):
            st.session_state.about_text = load_about_doc(about_file)
            st.session_state.about_name = about_file.name
            st.session_state["_about_file_id"] = about_file.file_id

    with col3:
        st.markdown('<div class="setup-card"><h4>3. Test Questions</h4><p>The questions you want to test the AI with, covering normal, tricky, and edge cases.</p></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload test_questions.xlsx", type=["xlsx", "xls"], key="questions_upl", label_visibility="collapsed")
        if uploaded is not None and uploaded.file_id != st.session_state.get("_questions_file_id"):
            raw = pd.read_excel(uploaded)
            st.session_state.df = ensure_columns(raw)
            st.session_state["_questions_file_id"] = uploaded.file_id

        st.download_button(
            "Download Test Question Template",
            data=df_to_excel_bytes(blank_template()),
            file_name="test_questions_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with st.expander("Optional: have AI draft the questions for you"):
        st.markdown(
            "This would read your dataset and About-the-AI document, then draft a full set of test questions "
            "covering all 9 testing areas automatically. It's not connected yet because it needs a paid AI API key "
            "(from a provider like OpenAI or Anthropic) — a small cost, a few cents for a full question set. "
            "Say the word and it can be wired up. "
            "For now, write your questions yourself using the template above, or fill in the Source column with "
            "'Dataset' / 'About the AI' / 'General' to mark where each question came from."
        )

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

if active_tab == TAB_LABELS[1]:
    if st.session_state.df is None:
        st.info("Upload `test_questions.xlsx` in the Setup tab first.")
    else:
        st.subheader("Send Questions to the AI")
        st.markdown(
            "Connect the AI agent's API below to send every question automatically, or just paste answers "
            "into the **AI Answer** column by hand — either works, and you can mix both."
        )

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
            response_path = st.text_input(
                "Where's the answer in the response? (dot path, e.g. answer or data.reply)",
                value="answer",
                key="api_response_path",
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
                            raw = call_agent_api(api_url, headers, payload, auth_type, auth_value)
                            answer = resolve_path(raw, response_path)
                            st.success("Connected. Here's what came back:")
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
                                raw = call_agent_api(api_url, headers, payload, auth_type, auth_value)
                                answer = resolve_path(raw, response_path)
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
                        if "editor_qa" in st.session_state:
                            del st.session_state["editor_qa"]
                        st.rerun()

        edited = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_qa",
            column_config={
                "Source": st.column_config.SelectboxColumn(options=SOURCE_OPTIONS),
            },
        )
        st.session_state.df = ensure_columns(edited)

if active_tab == TAB_LABELS[2]:
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
            "your judgment, using the Reviewer Score and Notes columns, and the dedicated Bias tab."
        )

        if st.button("Run automatic scoring", type="primary"):
            work = st.session_state.df.copy()
            ratios, auto_scores = [], []
            for _, row in work.iterrows():
                answered = str(row["AI Answer"]).strip() != ""
                r = similarity_score(row["Expected Answer"], row["AI Answer"]) if answered else 0.0
                ratios.append(round(r * 100, 1))
                auto_scores.append(to_rubric_score(r, answered))
            work["Similarity %"] = ratios
            work["Auto Score (0-3)"] = auto_scores
            if "Reviewer Score" not in work.columns:
                work["Reviewer Score"] = work["Auto Score (0-3)"]
            st.session_state.scored_df = work
            if "editor_scores" in st.session_state:
                del st.session_state["editor_scores"]

        if "scored_df" in st.session_state:
            scored = st.session_state.scored_df
            answered_mask = scored["Auto Score (0-3)"] != -1

            c1, c2, c3 = st.columns(3)
            c1.metric("Total questions", len(scored))
            c2.metric("Answered", int(answered_mask.sum()))
            avg_score = scored.loc[answered_mask, "Reviewer Score"].mean() if answered_mask.any() else 0
            c3.metric("Average score (0-3)", f"{avg_score:.2f}")

            st.markdown("**Review and correct scores below** (0 = No match, 3 = Full match):")
            reviewed = st.data_editor(
                scored,
                use_container_width=True,
                key="editor_scores",
                column_config={
                    "Reviewer Score": st.column_config.NumberColumn(min_value=0, max_value=3, step=1),
                },
                disabled=["ID", "Category", "Source", "Question", "Expected Answer", "AI Answer", "Similarity %", "Auto Score (0-3)"],
            )
            st.session_state.scored_df = reviewed

            st.download_button(
                "Download scored spreadsheet",
                data=df_to_excel_bytes(reviewed),
                file_name="all_answers_scored.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Click 'Run automatic scoring' to grade the answers you've collected so far.")

if active_tab == TAB_LABELS[3]:
    st.subheader("Detect Bias")
    st.markdown(
        "Bias means the AI performs noticeably worse for certain topics, question types, or phrasings than others. "
        "We check by comparing average scores across your question categories and sources — a category that scores "
        "much lower than the rest is a bias signal worth investigating."
    )
    if "scored_df" not in st.session_state:
        st.info("Run scoring in tab 3 (Evaluate the Answers) first.")
    else:
        scored = st.session_state.scored_df
        valid = scored[scored["Reviewer Score"] >= 0]
        if valid.empty:
            st.warning("No scored, answered questions yet.")
        else:
            overall_avg = valid["Reviewer Score"].mean()

            st.markdown("### By category")
            by_cat = valid.groupby("Category")["Reviewer Score"].mean().sort_values(ascending=False)
            st.bar_chart(by_cat, color="#7C3AED")

            st.markdown(f"**Overall average score:** {overall_avg:.2f} / 3")
            gap_threshold = 0.75
            flagged = by_cat[by_cat <= overall_avg - gap_threshold]
            if not flagged.empty:
                st.markdown("### Categories performing notably worse")
                for cat, score in flagged.items():
                    st.markdown(
                        f'<div class="flag-box"><b>{cat}</b>: {score:.2f} / 3 '
                        f"— {overall_avg - score:.2f} points below the overall average. "
                        "Possible bias or a gap in this topic.</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No category is significantly below the overall average.")

            st.markdown("### By source (Dataset vs About the AI vs General)")
            by_source = valid.groupby("Source")["Reviewer Score"].mean().reindex(SOURCE_OPTIONS).dropna()
            st.bar_chart(by_source, color="#7C3AED")
            st.caption(
                "A low 'About the AI' score means the agent isn't doing what it's described to do. "
                "A low Dataset score means its answers don't match what it was trained on."
            )

if active_tab == TAB_LABELS[4]:
    st.subheader("Conclusion & Analysis")
    st.markdown("This is the deliverable — everything else was in service of this.")
    if "scored_df" not in st.session_state:
        st.info("Run scoring in tab 3 (Evaluate the Answers) first.")
    else:
        scored = st.session_state.scored_df
        valid = scored[scored["Reviewer Score"] >= 0]

        agent_desc = st.text_area("What is this AI agent supposed to do?", height=80)
        conclusion_notes = st.text_area(
            "Your conclusion / recommendations (what needs to improve, is it safe to use as-is?)",
            height=100,
        )

        if st.button("Generate conclusion report", type="primary"):
            overall_avg = valid["Reviewer Score"].mean() if not valid.empty else 0
            by_cat = valid.groupby("Category")["Reviewer Score"].mean().sort_values(ascending=False) if not valid.empty else pd.Series(dtype=float)
            by_source = valid.groupby("Source")["Reviewer Score"].mean() if not valid.empty else pd.Series(dtype=float)

            lines = []
            lines.append("AI TESTER — CONCLUSION & ANALYSIS REPORT")
            lines.append(f"Generated: {date.today().isoformat()}")
            lines.append("")
            lines.append("1. WHAT WE TESTED")
            lines.append(agent_desc or "(not provided)")
            lines.append("")
            lines.append("2. REFERENCE MATERIALS USED")
            lines.append(f"  Dataset: {st.session_state.dataset_name or '(none provided)'}")
            lines.append(f"  About the AI document: {st.session_state.about_name or '(none provided)'}")
            lines.append("")
            lines.append("3. SUMMARY")
            lines.append(f"Questions tested: {len(scored)}")
            lines.append(f"Answered: {int((scored['AI Answer'].astype(str).str.strip() != '').sum())}")
            lines.append(f"Overall average score: {overall_avg:.2f} / 3")
            lines.append("")
            lines.append("4. SCORES BY CATEGORY")
            for cat, score in by_cat.items():
                lines.append(f"  - {cat}: {score:.2f} / 3")
            lines.append("")
            lines.append("5. SCORES BY SOURCE")
            for src, score in by_source.items():
                lines.append(f"  - {src}: {score:.2f} / 3")
            lines.append("")
            lines.append("6. BIAS FINDINGS")
            if not by_cat.empty:
                flagged = by_cat[by_cat <= overall_avg - 0.75]
                if not flagged.empty:
                    for cat, score in flagged.items():
                        lines.append(f"  - {cat} underperforms the average by {overall_avg - score:.2f} points.")
                else:
                    lines.append("  - No category performed significantly worse than average.")
            lines.append("")
            lines.append("7. CONCLUSION & RECOMMENDATIONS")
            lines.append(conclusion_notes or "(not provided)")
            lines.append("")
            lines.append(
                "NOTE: Scores are based on free text-similarity matching plus reviewer corrections. This covers "
                "Functional Correctness, Model Accuracy, and Bias & Fairness directly. Reliability (ask the same "
                "question twice and compare) and Explainability (can you understand how it got there) should be "
                "reviewed manually using the Notes column and added to this report."
            )

            report_text = "\n".join(lines)
            st.text_area("Report preview", report_text, height=400)
            st.download_button(
                "Download report (.txt)",
                data=report_text.encode("utf-8"),
                file_name="assessment_report.txt",
                mime="text/plain",
            )
