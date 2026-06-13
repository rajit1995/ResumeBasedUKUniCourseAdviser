"""
app.py

A Gradio web app that:
  1. Takes a resume (PDF / TXT upload, or pasted text).
  2. Extracts the candidate's skills / qualifications / interests.
  3. Uses a FAISS vector store (built from courses_dataset.json via
     build_vectorstore.py) to retrieve the most relevant UK courses
     (Retrieval-Augmented Generation).
  4. Asks a local Ollama LLM to pick and justify the top 10 course
     recommendations from the retrieved candidates.

Prerequisites
-------------
1. Install Ollama: https://ollama.com
2. Pull the models used here (you can change the names below):
       ollama pull nomic-embed-text
       ollama pull llama3
3. Generate the dataset and build the index (one-time setup):
       python generate_dataset.py
       python build_vectorstore.py

Run
---
    python app.py

Then open the printed local URL in your browser.
"""

import os
import json

import gradio as gr
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3")
INDEX_DIR = os.environ.get("INDEX_DIR", "faiss_index")
DATASET_PATH = os.environ.get("DATASET_PATH", "courses_dataset.json")
TOP_K_RETRIEVE = 25   # how many candidate courses to retrieve for the LLM to choose from
TOP_N_RECOMMEND = 10  # how many courses the LLM should finally recommend


# ---------------------------------------------------------------------------
# Vector store / LLM setup
# ---------------------------------------------------------------------------

def record_to_document(rec: dict) -> Document:
    focus_text = ", ".join(rec.get("focus", []))
    content = (
        f"Course: {rec['course']}\n"
        f"University: {rec['university']}\n"
        f"Focus areas: {focus_text}"
    )
    return Document(
        page_content=content,
        metadata={
            "university": rec["university"],
            "course": rec["course"],
            "focus": rec.get("focus", []),
        },
    )


def load_or_build_vectorstore(embeddings: OllamaEmbeddings) -> FAISS:
    """Load a previously-built FAISS index, or build a small one on the fly
    from the dataset if no index exists yet (so the app still runs even
    before `build_vectorstore.py` has been run on the full 100k dataset)."""

    if os.path.isdir(INDEX_DIR) and os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print(f"Loading existing FAISS index from '{INDEX_DIR}/' ...")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    print(f"No FAISS index found at '{INDEX_DIR}/'.")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Could not find '{DATASET_PATH}'. Run `python generate_dataset.py` first."
        )

    print(f"Building a quick index from a sample of '{DATASET_PATH}' "
          f"(run build_vectorstore.py separately to index the full dataset)...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # For an on-the-fly build, cap the sample size so startup stays fast.
    sample = records[:2000]
    docs = [record_to_document(r) for r in sample]
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved a starter index ({len(docs)} docs) to '{INDEX_DIR}/'.")
    return vectorstore


print("Initialising embeddings and LLM (this requires Ollama to be running)...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
vectorstore = load_or_build_vectorstore(embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a career and education advisor specialising in UK postgraduate "
            "courses. You will be given a candidate's resume text and a list of "
            "candidate UK courses retrieved from a course database. Your job is to "
            "select and rank the top {top_n} courses from the provided list that best "
            "match the candidate's qualifications, skills, and interests.\n\n"
            "Rules:\n"
            "- Only recommend courses that appear in the 'Candidate courses' list below. "
            "Do not invent universities or courses.\n"
            "- Rank from most relevant (1) to least relevant ({top_n}).\n"
            "- For each recommendation, give a short (1-2 sentence) reason linking it to "
            "specific elements of the resume (skills, projects, degree background, etc.).\n"
            "- Respond in clean Markdown using a numbered list, with the format:\n"
            "  **N. Course Name — University Name**\n"
            "  Reason: ...\n",
        ),
        (
            "human",
            "Resume text:\n"
            "-----\n"
            "{resume_text}\n"
            "-----\n\n"
            "Candidate courses (retrieved from the database):\n"
            "-----\n"
            "{candidates}\n"
            "-----\n\n"
            "Now produce the top {top_n} recommended courses as instructed.",
        ),
    ]
)

rag_chain = RECOMMENDATION_PROMPT | llm


# ---------------------------------------------------------------------------
# Resume text extraction
# ---------------------------------------------------------------------------

def extract_text_from_file(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------------

def build_candidates_text(docs) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        focus = ", ".join(meta.get("focus", []))
        lines.append(f"{i}. {meta['course']} — {meta['university']} (Focus: {focus})")
    return "\n".join(lines)


def recommend_courses(resume_file, resume_text_input):
    # Resolve resume text from either an uploaded file or the textbox
    resume_text = ""
    if resume_file is not None:
        try:
            resume_text = extract_text_from_file(resume_file.name if hasattr(resume_file, "name") else resume_file)
        except Exception as e:
            return f"Error reading resume file: {e}"
    elif resume_text_input and resume_text_input.strip():
        resume_text = resume_text_input.strip()
    else:
        return "Please upload a resume file (PDF/TXT) or paste your resume text."

    if not resume_text.strip():
        return "Could not extract any text from the provided resume."

    # Truncate very long resumes to keep prompts manageable
    truncated_resume = resume_text[:6000]

    # Retrieve relevant courses (RAG step)
    retrieved_docs = retriever.invoke(truncated_resume)
    if not retrieved_docs:
        return "No matching courses were found in the database."

    candidates_text = build_candidates_text(retrieved_docs)

    # Generate the final ranked recommendations
    response = rag_chain.invoke(
        {
            "resume_text": truncated_resume,
            "candidates": candidates_text,
            "top_n": TOP_N_RECOMMEND,
        }
    )

    return response.content


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="UK Course Recommender (RAG + Ollama)") as demo:
    gr.Markdown(
        "# 🎓 UK Course Recommender\n"
        "Upload your resume (PDF or TXT) **or** paste its text below. "
        "The app retrieves the most relevant UK postgraduate courses from a "
        "course database using embeddings, then asks a local LLM (via Ollama) "
        f"to suggest the top {TOP_N_RECOMMEND} courses based on your qualifications."
    )

    with gr.Row():
        with gr.Column():
            resume_file = gr.File(label="Upload resume (PDF or TXT)", file_types=[".pdf", ".txt"])
            resume_text_input = gr.Textbox(
                label="...or paste your resume text here",
                lines=12,
                placeholder="Paste resume text if you don't want to upload a file.",
            )
            submit_btn = gr.Button("Get Course Recommendations", variant="primary")

        with gr.Column():
            output = gr.Markdown(label="Recommended Courses")

    submit_btn.click(
        fn=recommend_courses,
        inputs=[resume_file, resume_text_input],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()
