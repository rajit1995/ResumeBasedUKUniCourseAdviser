
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
DATASET_PATH = os.environ.get("DATASET_PATH", "courses_dataset.jsonl")
TOP_K_RETRIEVE = 60   # how many candidate courses to retrieve before level-filtering
MAX_CANDIDATES = 40   # cap on filtered candidates passed to the LLM
TOP_N_RECOMMEND = 10  # how many courses the LLM should finally recommend

# Degree prefixes (the first "word" of a course title) used to classify
# each course as undergraduate or postgraduate, so recommendations can be
# restricted to the level appropriate for the candidate.
UNDERGRAD_DEGREES = {"BSc", "BA", "BEng", "LLB", "MBBS"}
POSTGRAD_DEGREES = {"MSc", "MA", "MEng", "MRes", "MPhil", "MBA", "PGDip", "PGCert", "LLM"}


def course_level(course: str) -> str:
    """Classify a course title as 'undergraduate', 'postgraduate', or
    'unknown' based on its leading degree abbreviation."""
    degree = course.split()[0] if course else ""
    if degree in POSTGRAD_DEGREES:
        return "postgraduate"
    if degree in UNDERGRAD_DEGREES:
        return "undergraduate"
    return "unknown"


# ---------------------------------------------------------------------------
# Vector store / LLM setup
# ---------------------------------------------------------------------------

def record_to_document(rec: dict) -> Document:
    """Must match record_to_document in build_vectorstore.py exactly, so
    that the index built there is compatible with this app, and so any
    on-the-fly starter index built here matches the same shape."""
    focus_text = ", ".join(rec.get("focus", []))
    content = (
        f"Course: {rec['course']}\n"
        f"Department: {rec.get('department', '')}\n"
        f"University: {rec['university']}\n"
        f"Focus areas: {focus_text}"
    )
    return Document(
        page_content=content,
        metadata={
            "university": rec["university"],
            "course": rec["course"],
            "department": rec.get("department", ""),
            "focus": rec.get("focus", []),
            "region": rec.get("region"),
            "founded_year": rec.get("founded_year"),
            "motto": rec.get("motto"),
            "uk_rank": rec.get("uk_rank"),
            "acceptance_rate_pct": rec.get("acceptance_rate_pct"),
            "international_acceptance_rate_pct": rec.get("international_acceptance_rate_pct"),
            "india_friendly_score": rec.get("india_friendly_score"),
            "competitiveness": rec.get("competitiveness"),
        },
    )


def _load_flat_records(dataset_path):
    """Load the flattened per-course dataset (.jsonl or .json)."""
    if dataset_path.endswith(".jsonl"):
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_build_vectorstore(embeddings: OllamaEmbeddings) -> FAISS:
    """Load a previously-built FAISS index, or build a small one on the fly
    from the dataset if no index exists yet (so the app still runs even
    before `build_vectorstore.py` has been run on the full dataset)."""

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
    records = _load_flat_records(DATASET_PATH)

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
            "You are an admissions-aware career and education advisor specialising "
            "in UK university courses, with particular experience advising Indian "
            "and other international applicants.\n\n"
            "You will be given a candidate's resume text and a list of candidate "
            "UK university courses retrieved from a course database. Each candidate "
            "course includes:\n"
            "  - the course name and department\n"
            "  - the university name, region and UK rank\n"
            "  - the university's overall acceptance rate\n"
            "  - the university's acceptance rate for international applicants\n"
            "  - an India-friendliness score (1-10), reflecting how accessible / "
            "welcoming the university is reported to be for Indian applicants "
            "(scholarships, dedicated recruitment, visa and post-study work support, "
            "size of existing Indian student community)\n"
            "  - a competitiveness rating (Very High / High / Medium / Low to Medium)\n\n"
            "IMPORTANT — Study level: {level_instruction}\n\n"
            "Your job is to select and rank the top {top_n} courses from the provided "
            "list that the candidate is REALISTICALLY ELIGIBLE FOR — i.e. courses "
            "that both (a) genuinely match the candidate's qualifications, skills, "
            "academic background and interests, AND (b) have an admission "
            "likelihood that is reasonable given the candidate's profile, the "
            "university's UK rank/competitiveness, its acceptance rates, and (where "
            "relevant) its India-friendliness score and openness to international "
            "applicants.\n\n"
            "Rules:\n"
            "- Only recommend courses that appear in the 'Candidate courses' list "
            "below. Do not invent universities, departments or courses.\n"
            "- Every candidate course listed has a 'Level' tag (Undergraduate or "
            "Postgraduate). Only recommend courses whose Level matches the study "
            "level specified above — do not recommend courses from the other "
            "level under any circumstances.\n"
            "- Favour a realistic, achievable mix: don't recommend only the most "
            "prestigious / Very High competitiveness universities unless the "
            "candidate's profile is exceptionally strong. Where appropriate, "
            "include a healthy spread across competitiveness levels (e.g. some "
            "ambitious 'reach' options alongside solid 'match' and safer 'likely' "
            "options) so the candidate has realistic choices.\n"
            "- Rank from most relevant/realistic (1) to least (({top_n})).\n"
            "- For each recommendation, give a short (2-3 sentence) reason that "
            "covers: (i) why the course fits the candidate's background, skills or "
            "interests, and (ii) why admission is realistic for this candidate, "
            "referencing the university's ranking/competitiveness, acceptance rate, "
            "and India-friendliness/international acceptance where relevant.\n"
            "- Respond in clean Markdown using a numbered list, with the format:\n"
            "  **N. Course Name — University Name (UK Rank #X, Competitiveness: Y)**\n"
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
            "Now produce the top {top_n} recommended courses/universities as "
            "instructed, focusing on what this candidate is realistically "
            "eligible for.",
        ),
    ]
)

rag_chain = RECOMMENDATION_PROMPT | llm


# ---------------------------------------------------------------------------
# Education-stage classification (Bachelor's completed vs. not yet)
# ---------------------------------------------------------------------------

EDUCATION_STAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You read a candidate's analysis report / resume text and determine "
            "their current education stage with respect to a Bachelor's degree "
            "(undergraduate degree).\n\n"
            "Respond with EXACTLY one word, nothing else:\n"
            "- 'completed' — the candidate has already completed (graduated "
            "from) a Bachelor's degree, or holds an equivalent or higher "
            "qualification (e.g. Master's, MBBS, PhD).\n"
            "- 'not_completed' — the candidate has NOT yet completed a "
            "Bachelor's degree. This includes candidates who are currently "
            "pursuing a Bachelor's degree, are about to start one, or are "
            "still in school (e.g. high school / 12th grade).\n\n"
            "If the text is ambiguous or doesn't mention education clearly, "
            "respond 'not_completed'.",
        ),
        (
            "human",
            "Analysis report / resume text:\n-----\n{resume_text}\n-----\n\n"
            "Education stage (one word: completed / not_completed):",
        ),
    ]
)

stage_chain = EDUCATION_STAGE_PROMPT | llm


# ---------------------------------------------------------------------------
# Resume analysis (resume -> analysis report)
# ---------------------------------------------------------------------------

RESUME_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an experienced admissions counsellor's assistant. You will be "
            "given the raw text extracted from a candidate's resume/CV. Read it "
            "thoroughly and produce a DETAILED ANALYSIS REPORT about the "
            "candidate, in clean Markdown, using the following sections. Be "
            "thorough and specific - pull out concrete details (names, dates, "
            "grades, technologies, durations, outcomes) rather than vague "
            "generalisations. Where useful, use sub-bullets to expand on an "
            "item (e.g. what a project involved, what a role's responsibilities "
            "were).\n\n"
            "## 1. Education\n"
            "For EACH qualification listed (school, undergraduate, postgraduate, "
            "etc.), give its own sub-section or bullet with: institution name, "
            "degree/qualification name, field/major and any specialisation or "
            "minor, start and end dates (or 'ongoing'), grade/GPA/percentage if "
            "mentioned, and any notable coursework, honours, or academic awards. "
            "Then add a clearly-labelled paragraph, 'Current education stage:', "
            "stating explicitly and unambiguously ONE of the following, with "
            "supporting detail:\n"
            "  - 'Bachelor's degree completed' - name the exact degree, field, "
            "institution and completion year.\n"
            "  - 'Currently pursuing a Bachelor's degree' - name the degree, "
            "field, institution, current year/semester, and expected graduation "
            "year if stated.\n"
            "  - 'Has not yet started a Bachelor's degree' - describe the "
            "candidate's current level (e.g. completed/ongoing secondary "
            "schooling, grade/standard, board, expected completion year).\n\n"
            "## 2. Technical Skills\n"
            "Group into categories as relevant (e.g. Programming Languages, "
            "Frameworks & Libraries, Tools & Platforms, Data/ML, Cloud & DevOps, "
            "Databases, Design, etc.), listing every specific skill/technology "
            "mentioned and, where evident, the candidate's apparent proficiency "
            "or how extensively it was used.\n\n"
            "## 3. Soft Skills & Transferable Strengths\n"
            "Identify soft skills (e.g. leadership, communication, teamwork, "
            "problem-solving, time management) and briefly cite the evidence "
            "from the resume that supports each one (e.g. a leadership role, a "
            "competition win, a group project).\n\n"
            "## 4. Work Experience & Internships\n"
            "For each role: job title, organisation, duration, and a bullet "
            "breakdown of key responsibilities and quantifiable achievements or "
            "impact (numbers, percentages, scale, outcomes) wherever the resume "
            "provides them.\n\n"
            "## 5. Projects\n"
            "For each significant project: name/title, a short description of "
            "the problem/goal, the technologies or methods used, the "
            "candidate's specific role/contribution, and the outcome or result.\n\n"
            "## 6. Certifications, Publications & Achievements\n"
            "List any certifications (with issuing body and date if given), "
            "publications, competition wins, scholarships, or other notable "
            "achievements.\n\n"
            "## 7. Extracurricular Activities & Leadership\n"
            "Clubs, societies, volunteering, sports, leadership positions, and "
            "what they involved.\n\n"
            "## 8. Interests & Career Goals\n"
            "Any stated or strongly implied academic interests, preferred "
            "subject areas, and short/long-term career goals - including any "
            "mentioned preference for studying abroad, the UK specifically, or "
            "particular countries.\n\n"
            "## 9. Overall Assessment\n"
            "A substantive paragraph (4-6 sentences) summarising the candidate's "
            "academic strengths, technical depth, breadth vs. specialisation, "
            "standout achievements, any gaps or areas that may need "
            "strengthening, and an overall view of their readiness and "
            "suitability for further study (Bachelor's or Master's level, as "
            "appropriate to their current stage).\n\n"
            "Base everything strictly on the resume text provided - do not "
            "invent details. If a section genuinely has no information, write "
            "'Not specified' for that section rather than omitting it.",
        ),
        (
            "human",
            "Resume text:\n-----\n{resume_text}\n-----\n\nProduce the detailed analysis report.",
        ),
    ]
)

analysis_chain = RESUME_ANALYSIS_PROMPT | llm


def analyze_and_recommend(resume_file):
    """Single-button workflow: extract the resume, generate a detailed
    analysis report, then immediately feed that report into the course
    recommendation pipeline. Returns (analysis_report, recommendations)."""
    if resume_file is None:
        return (
            "Please upload a resume (PDF or TXT) first.",
            "",
        )

    try:
        resume_text = extract_text_from_file(resume_file.name if hasattr(resume_file, "name") else resume_file)
    except Exception as e:
        return f"Error reading file: {e}", ""

    if not resume_text.strip():
        return "Could not extract any text from the uploaded resume.", ""

    truncated_resume = resume_text[:8000]

    analysis_response = analysis_chain.invoke({"resume_text": truncated_resume})
    analysis_report = analysis_response.content

    recommendations = recommend_courses(analysis_report)

    return analysis_report, recommendations


def determine_target_level(resume_text: str) -> tuple[str, str]:
    """Returns (target_level, stage_label) where target_level is
    'postgraduate' or 'undergraduate', and stage_label is a short
    human-readable explanation of the detected stage."""
    try:
        response = stage_chain.invoke({"resume_text": resume_text})
        stage = (response.content or "").strip().lower()
    except Exception:
        stage = ""

    if "not_completed" in stage:
        return "undergraduate", "has not yet completed a Bachelor's degree"
    if "completed" in stage:
        return "postgraduate", "has already completed a Bachelor's degree (or higher)"

    # Default to undergraduate if the classifier response was unclear -
    # i.e. only show Bachelor's-level courses unless we're confident a
    # Bachelor's degree has been completed.
    return "undergraduate", "education stage unclear — defaulting to Bachelor's-level courses"


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

        rank = meta.get("uk_rank")
        rank_str = f"#{rank}" if rank is not None else "Unranked"

        acceptance = meta.get("acceptance_rate_pct")
        intl_acceptance = meta.get("international_acceptance_rate_pct")
        india_score = meta.get("india_friendly_score")
        competitiveness = meta.get("competitiveness", "Unknown")
        region = meta.get("region", "")

        lines.append(
            f"{i}. {meta['course']} ({meta.get('department', '')}) — "
            f"{meta['university']} ({region}, UK Rank {rank_str}) | "
            f"Level: {course_level(meta['course']).capitalize()} | "
            f"Competitiveness: {competitiveness} | "
            f"Acceptance rate: {acceptance}% | "
            f"International acceptance rate: {intl_acceptance}% | "
            f"India-friendliness (1-10): {india_score} | "
            f"Focus: {focus}"
        )
    return "\n".join(lines)


def recommend_courses(analysis_report_input):
    if not analysis_report_input or not analysis_report_input.strip():
        return (
            "Please upload a resume and click **Analyze & Recommend** to generate "
            "an analysis report and recommendations."
        )

    resume_text = analysis_report_input.strip()

    # Truncate very long input to keep prompts manageable
    truncated_resume = resume_text[:8000]

    # Determine whether the candidate has completed a Bachelor's degree.
    # - completed       -> recommend Master's and above (postgraduate)
    # - not completed   -> recommend Bachelor's-level courses only (undergraduate)
    target_level, stage_label = determine_target_level(truncated_resume)

    if target_level == "postgraduate":
        level_instruction = (
            "The candidate has already completed a Bachelor's degree (or higher). "
            "Only recommend POSTGRADUATE courses (Master's and above — e.g. MSc, "
            "MA, MEng, MRes, MPhil, MBA, LLM, PGDip, PGCert). Do NOT recommend any "
            "undergraduate (Bachelor's) courses."
        )
    else:
        level_instruction = (
            "The candidate has NOT yet completed a Bachelor's degree (they may be "
            "currently studying towards one, about to start one, or still in "
            "school). Only recommend UNDERGRADUATE / Bachelor's-level courses "
            "(e.g. BSc, BA, BEng, LLB, MBBS). Do NOT recommend any postgraduate "
            "(Master's and above) courses."
        )

    # Retrieve relevant courses (RAG step)
    retrieved_docs = retriever.invoke(truncated_resume)
    if not retrieved_docs:
        return "No matching courses were found in the database."

    # Restrict to courses at the appropriate level for this candidate.
    level_filtered = [d for d in retrieved_docs if course_level(d.metadata["course"]) == target_level]
    if not level_filtered:
        # Fall back to the unfiltered set rather than returning nothing,
        # but the prompt instruction will still steer the LLM toward the
        # right level if any matching courses exist in it.
        level_filtered = retrieved_docs

    candidates_text = build_candidates_text(level_filtered[:MAX_CANDIDATES])

    # Generate the final ranked recommendations
    response = rag_chain.invoke(
        {
            "resume_text": truncated_resume,
            "candidates": candidates_text,
            "top_n": TOP_N_RECOMMEND,
            "level_instruction": level_instruction,
        }
    )

    level_note = (
        f"_Detected education stage: the candidate {stage_label}. "
        f"Showing **{target_level}** course recommendations accordingly._\n\n"
    )
    return level_note + response.content


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="UK Course Recommender (RAG + Ollama)") as demo:
    gr.Markdown(
        "# 🎓 UK Course Recommender\n"
        "Upload a resume (PDF or TXT) and click **Analyze Resume** - a local "
        "LLM (via Ollama) will read it and write an **Analysis Report** "
        "(education, skills, experience, interests). Review/edit the report if "
        "needed, then click **Get Course Recommendations**.\n\n"
        "The app retrieves the most relevant UK university courses from a "
        "course database using embeddings, then asks the LLM to suggest the "
        f"top {TOP_N_RECOMMEND} courses/universities you're realistically "
        "eligible for - based on your qualifications and interests, "
        "course/university ranking, competitiveness, acceptance rates, and how "
        "welcoming each university is towards international (including Indian) "
        "applicants.\n\n"
        "**Study level is detected automatically from the analysis report:** if "
        "it shows the candidate has already completed a Bachelor's degree, only "
        "Master's-and-above courses are recommended; if the candidate hasn't "
        "completed a Bachelor's degree yet (including currently studying for "
        "one), only Bachelor's-level courses are recommended."
    )

with gr.Blocks(title="UK Course Recommender (RAG + Ollama)") as demo:
    gr.Markdown(
        "# 🎓 UK Course Recommender\n"
        "Upload a resume (PDF or TXT) and click **Analyze & Recommend**. A "
        "local LLM (via Ollama) will:\n"
        "1. Read the resume and write a detailed **Analysis Report** "
        "(education, skills, experience, projects, interests).\n"
        "2. Use that report to determine the candidate's study level and "
        "retrieve the most relevant UK university courses from the course "
        "database.\n"
        f"3. Recommend the top {TOP_N_RECOMMEND} courses/universities the "
        "candidate is realistically eligible for - based on their "
        "qualifications and interests, course/university ranking, "
        "competitiveness, acceptance rates, and how welcoming each university "
        "is towards international (including Indian) applicants.\n\n"

    )

    with gr.Row():
        with gr.Column():
            resume_file = gr.File(label="Upload resume (PDF or TXT)", file_types=[".pdf", ".txt"])
            with gr.Row():
                analyze_btn = gr.Button("🚀 Analyze & Recommend", variant="primary")
                refresh_btn = gr.Button("🔄 Refresh")
            analysis_report_input = gr.Textbox(
                label="Analysis Report (auto-generated)",
                lines=14,
                placeholder=(
                    "Upload a resume and click 'Analyze & Recommend' - the "
                    "detailed analysis report will appear here."
                ),
            )

        with gr.Column():
            output = gr.Markdown(label="Recommended Courses")

    analyze_btn.click(
        fn=analyze_and_recommend,
        inputs=resume_file,
        outputs=[analysis_report_input, output],
    )

    refresh_btn.click(
        fn=lambda: (None, "", ""),
        inputs=None,
        outputs=[resume_file, analysis_report_input, output],
    )


if __name__ == "__main__":
    demo.launch()
