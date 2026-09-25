import re
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

PDF_PATH = Path(__file__).parent / "Meridian_Global_Logistics_Handbook.pdf"
EMPLOYEE_FILE = Path(__file__).parent / "Employee_Master.xlsx"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

EMP_ID_PATTERN = re.compile(r"\bMGL\d{5}\b", re.IGNORECASE)
LEAVE_REQUEST_PATTERN = re.compile(r"\b(\d+)\s*day", re.IGNORECASE)

LEAVE_TYPE_COLUMNS = {
    "casual": "Casual_Leave_Balance",
    "sick": "Sick_Leave_Balance",
    "earned": "Earned_Leave_Balance",
}

st.set_page_config(
    page_title="Meridian Employee Assistant",
    page_icon="📦",
    layout="centered"
)
st.title("📦 Meridian Employee Assistant")
st.caption("Your AI assistant for Meridian Global Logistics employee policies")


@st.cache_resource(show_spinner="Loading the document and building the search index...")
def load_knowledge_base():
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))
    embedder = SentenceTransformer(EMBED_MODEL)

    doc = pymupdf.open(PDF_PATH)
    chunks = []
    for page in doc:
        text = page.get_text()
        for i in range(0, len(text), 450):
            chunk = text[i:i + 500].strip()
            if chunk:
                chunks.append(chunk)

    chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True)
    employees_df = pd.read_excel(EMPLOYEE_FILE)

    return client, embedder, chunks, chunk_embeddings, employees_df


client, embedder, chunks, chunk_embeddings, employees_df = load_knowledge_base()


def find_hr_contact(location: str):
    """Find an HR department person at the same location, as the notified rep."""
    hr_rows = employees_df[
        (employees_df["Department"] == "Human Resources")
        & (employees_df["Location"] == location)
    ]
    if not hr_rows.empty:
        rep = hr_rows.iloc[0]
        return rep["Full_Name"], rep["Email"]
    return "the HR team", "people@meridianlogistics.com"


def lookup_employee(emp_id: str):
    match = employees_df[employees_df["Employee_ID"].astype(str).str.upper() == emp_id]
    return None if match.empty else match.iloc[0]


def handle_message(user_text: str) -> str:
    """Rule-based handling for employee ID lookups and leave applications;
    falls back to the RAG + LLM flow for everything else."""

    # Remember the last employee ID mentioned in this session, so the person
    # doesn't have to repeat it in follow-up messages.
    id_match = EMP_ID_PATTERN.search(user_text)
    if id_match:
        st.session_state.current_emp_id = id_match.group(0).upper()

    emp_id = st.session_state.get("current_emp_id")

    # --- Intent: apply for leave ("apply for 2 days casual leave") ---
    day_match = LEAVE_REQUEST_PATTERN.search(user_text)
    leave_type_key = next((k for k in LEAVE_TYPE_COLUMNS if k in user_text.lower()), None)

    if day_match and leave_type_key:
        if not emp_id:
            return "Please share your Employee ID first (e.g. MGL10001) so I can apply the leave."

        row = lookup_employee(emp_id)
        if row is None:
            return f"I couldn't find an employee with ID {emp_id}."

        days = int(day_match.group(1))
        leave_col = LEAVE_TYPE_COLUMNS[leave_type_key]
        balance = row[leave_col]

        if days > balance:
            return (
                f"You only have {balance} day(s) of {leave_type_key} leave left, "
                f"so I can't apply for {days} day(s)."
            )

        new_balance = balance - days
        employees_df.loc[row.name, leave_col] = new_balance
        employees_df.to_excel(EMPLOYEE_FILE, index=False)

        hr_name, hr_email = find_hr_contact(row["Location"])
        return (
            f"✅ Applied {days} day(s) of {leave_type_key} leave for {row['Full_Name']}. "
            f"Remaining {leave_type_key} balance: {new_balance} day(s). "
            f"{hr_name} ({hr_email}) has been informed of your leave, and a confirmation "
            f"email will be sent to {row['Email']}."
        )

    # --- Intent: just mentioned an employee ID, show balance ---
    if id_match:
        row = lookup_employee(emp_id)
        if row is None:
            return f"I couldn't find an employee with ID {emp_id}."
        return (
            f"Hi {row['Full_Name']}! Your leave balance — "
            f"Casual: {row['Casual_Leave_Balance']}d, "
            f"Sick: {row['Sick_Leave_Balance']}d, "
            f"Earned: {row['Earned_Leave_Balance']}d."
        )

    # --- Otherwise: normal RAG-grounded HR policy question ---
    question_embedding = embedder.encode([user_text], normalize_embeddings=True)[0]
    scores = chunk_embeddings @ question_embedding
    top_chunks = [chunks[i] for i in np.argsort(scores)[::-1][:TOP_K]]
    context = "\n\n".join(top_chunks)

    api_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_text}"},
    ]
    completion = client.chat.completions.create(model=LLM_MODEL, messages=api_messages)
    return completion.choices[0].message.content


system_prompt = (
    "You are the Meridian Global Logistics employee assistant. Answer questions "
    "using ONLY the provided handbook context. If the person greets you, reply "
    "with a friendly greeting. If the answer isn't in the context, politely say "
    "so and suggest contacting people@meridianlogistics.com, rather than saying "
    "a blunt 'I don't know'. Do not use outside knowledge."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about HR policy, share your Employee ID, or apply for leave")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = handle_message(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
