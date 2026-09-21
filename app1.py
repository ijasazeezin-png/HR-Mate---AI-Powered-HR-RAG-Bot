import os
import pandas as pd
import numpy as np
import pymupdf
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import re

EMPLOYEE_FILE = "Employee_Master.xlsx"
HANDBOOK_FILE = "Meridian_Global_Logistics_Handbook.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

st.set_page_config(page_title="Meridian Employee Assistant", page_icon="\U0001F4E6")


def get_api_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        key = None
    return key or os.environ.get("GROQ_API_KEY", "")


api_key = get_api_key()
if not api_key:
    st.error("No Groq API key found. Set GROQ_API_KEY as an environment variable or in secrets.toml.")
    st.stop()


@st.cache_resource(show_spinner="Loading employee records and company handbook...")
def load_everything(_api_key):
    client = Groq(api_key=_api_key)

    # --- Employee master data ---
    if not os.path.exists(EMPLOYEE_FILE):
        raise FileNotFoundError(f"Could not find '{EMPLOYEE_FILE}'. Place it next to this app.")
    employees = pd.read_excel(EMPLOYEE_FILE)
    employees["Employee_ID"] = employees["Employee_ID"].astype(str).str.upper()

    # --- Handbook RAG index ---
    if not os.path.exists(HANDBOOK_FILE):
        raise FileNotFoundError(f"Could not find '{HANDBOOK_FILE}'. Place it next to this app.")
    embedder = SentenceTransformer(EMBED_MODEL)
    doc = pymupdf.open(HANDBOOK_FILE)
    chunks, pages = [], []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        start = 0
        while start < len(text):
            chunk = text[start:start + 500].strip()
            if chunk:
                chunks.append(chunk)
                pages.append(page_num)
            start += 450
    chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True)

    return client, employees, embedder, chunks, pages, np.array(chunk_embeddings)


client, employees_df, embedder, chunks, pages, chunk_embeddings = load_everything(api_key)

EMP_ID_PATTERN = re.compile(r"\bMGL\d{5}\b", re.IGNORECASE)
LEAVE_TYPE_MAP = {
    "casual": "Casual_Leave_Balance",
    "cl": "Casual_Leave_Balance",
    "sick": "Sick_Leave_Balance",
    "sl": "Sick_Leave_Balance",
}
APPLY_KEYWORDS = ["apply", "book", "take", "want to take", "request"]


def retrieve_handbook_context(question: str, k: int = TOP_K):
    q_emb = embedder.encode([question], normalize_embeddings=True)[0]
    scores = chunk_embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], pages[i]) for i in top_idx]


def find_employee(emp_id: str):
    match = employees_df[employees_df["Employee_ID"] == emp_id.upper()]
    if match.empty:
        return None
    return match.iloc[0]


def format_employee_details(emp) -> str:
    return (
        f"**{emp['Full_Name']}** ({emp['Employee_ID']})\n\n"
        f"- Department: {emp['Department']}\n"
        f"- Designation: {emp['Designation']}\n"
        f"- Location: {emp['Location']}\n"
        f"- Date of Joining: {emp['Date_of_Joining']}\n"
        f"- Status: {emp['Employment_Status']}\n\n"
        f"**Leave Balance**\n"
        f"- Casual Leave: {emp['Casual_Leave_Balance']} days\n"
        f"- Sick Leave: {emp['Sick_Leave_Balance']} days\n"
        f"- Earned Leave: {emp['Earned_Leave_Balance']} days"
    )


def try_parse_leave_application(message: str):
    """Return (leave_type_column, num_days) if the message looks like a leave
    application for Casual or Sick leave, else None."""
    lower = message.lower()
    if not any(k in lower for k in APPLY_KEYWORDS):
        return None

    day_match = re.search(r"(\d+)\s*day", lower)
    if not day_match:
        return None
    num_days = int(day_match.group(1))

    for keyword, column in LEAVE_TYPE_MAP.items():
        if keyword in lower:
            return column, num_days
    return None


def handle_employee_message(emp_id: str, message: str) -> str:
    emp = find_employee(emp_id)
    if emp is None:
        return (
            f"I couldn't find an employee record for **{emp_id.upper()}**. "
            "Please double-check the ID and try again, or contact HR if the "
            "problem continues."
        )

    leave_request = try_parse_leave_application(message)
    if leave_request:
        column, num_days = leave_request
        leave_label = "Casual" if column == "Casual_Leave_Balance" else "Sick"
        current_balance = int(employees_df.loc[employees_df["Employee_ID"] == emp["Employee_ID"], column].iloc[0])

        if num_days > current_balance:
            return (
                f"You've requested {num_days} day(s) of {leave_label} Leave, but "
                f"{emp['Full_Name']} ({emp['Employee_ID']}) only has **{current_balance} "
                f"day(s)** of {leave_label} Leave remaining. Please adjust the request or "
                f"contact HR to discuss options."
            )

        # Deduct and persist in the shared in-memory table (see README for the
        # limitation this implies on Streamlit Cloud).
        new_balance = current_balance - num_days
        employees_df.loc[employees_df["Employee_ID"] == emp["Employee_ID"], column] = new_balance

        return (
            f"\u2705 Leave application submitted for **{emp['Full_Name']}** "
            f"({emp['Employee_ID']}).\n\n"
            f"- Leave type: {leave_label} Leave\n"
            f"- Days: {num_days}\n"
            f"- New {leave_label} Leave balance: **{new_balance} day(s)**\n\n"
            f"_Note: Earned Leave, Maternity, Paternity, and Bereavement Leave require "
            f"manager approval \u2014 please apply for those through the MGL Connect HR portal._"
        )

    # Not a leave application - just show the employee's details.
    return format_employee_details(emp)


# ----------------------------- Chat UI -----------------------------

st.title("\U0001F4E6 Meridian Employee Assistant")
st.caption(
    f"Ask a general company/policy question, or include your Employee ID "
    f"(e.g. MGL10001) to check your details or apply for Casual/Sick leave. "
    f"{len(employees_df)} employee records loaded."
)

SYSTEM_PROMPT = (
    "You are the Meridian Global Logistics employee assistant. Answer questions "
    "using ONLY the provided handbook context. If the person greets you, reply "
    "with a friendly greeting. If the answer isn't in the context, politely say "
    "so and suggest contacting people@meridianlogistics.com, rather than saying "
    "a blunt 'I don't know'. Do not use outside knowledge."
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Ask about company policy, or include your Employee ID...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        emp_id_match = EMP_ID_PATTERN.search(prompt)

        if emp_id_match:
            # Structured path: real data, no LLM involved in the numbers.
            response = handle_employee_message(emp_id_match.group(0), prompt)
            st.markdown(response)
        else:
            # RAG fallback for general company/policy questions.
            with st.spinner("Searching the handbook..."):
                results = retrieve_handbook_context(prompt)
                context = "\n\n".join(f"[Page {p}] {c}" for c, p in results)

                api_messages = st.session_state.messages[:-1] + [{
                    "role": "user",
                    "content": f"Handbook context:\n{context}\n\nQuestion: {prompt}",
                }]

                completion = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=api_messages,
                )
                response = completion.choices[0].message.content

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

