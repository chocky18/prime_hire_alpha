import os
import re
import json
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, Body, HTTPException
import uvicorn
import docx2txt
import pdfplumber
import uvicorn

# ---- Try to load secrets in multiple environments ----
try:
    import streamlit as st
    ST_MODE = True
except ImportError:
    ST_MODE = False

if ST_MODE and "OPENAI_API_KEY" in st.secrets:
    # ✅ Streamlit Cloud secrets
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "candidates-db")
else:
    # ✅ Local dev: use dotenv or GitHub Actions (os.environ)
    from dotenv import load_dotenv
    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "candidates-db")

# ---- Logging ----
logging.basicConfig(level=logging.DEBUG)

# ---- OpenAI & Pinecone ----
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ---- Clients ----
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ---- Ensure Pinecone index ----
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,  # text-embedding-3-large
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)


# ---- FastAPI App ----
app = FastAPI()


# -------- Helper Functions -------- #
def read_resume(file_path: str):
    """Extract text from PDF or DOCX."""
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logging.debug(f"[DEBUG] Extracted PDF text ({file_path}):\n{text[:500]}...\n")
        return text.strip()
    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path).strip()
        logging.debug(f"[DEBUG] Extracted DOCX text ({file_path}):\n{text[:500]}...\n")
        return text
    else:
        raise ValueError("Unsupported file type")


def chunk_text(text: str, chunk_size: int = 500):
    """Split text into smaller chunks (≈paragraphs)."""
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def get_embedding(text: str):
    """Generate OpenAI embedding."""
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return resp.data[0].embedding


def extract_metadata(text: str):
    """Extract structured candidate info from resume text using GPT (CoT + few-shot)."""
    chunks = chunk_text(text, chunk_size=500)
    metadata = {
        "name": None,
        "designation": None,
        "skills": [],
        "experience_years": 0,
        "location": None
    }

    few_shot_example = """
    Resume text:
    John Doe - Senior Backend Engineer
    Worked 6 years at Google (Mountain View, USA) specializing in Python, Django, and Kubernetes.

    Expected JSON output:
    {
      "name": "John Doe",
      "designation": "Senior Backend Engineer",
      "skills": ["Python", "Django", "Kubernetes"],
      "experience_years": 6,
      "location": "Mountain View, USA"
    }
    """

    for i, chunk in enumerate(chunks):
        prompt = f"""
        You are an expert resume parser.
        Use the resume text and extract metadata step by step before answering.
        Output ONLY valid JSON, no extra text.

        {few_shot_example}

        Resume text:
        {chunk}
        """

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw_response = resp.choices[0].message.content.strip()
            logging.debug(f"[DEBUG] LLM response for chunk {i}:\n{raw_response}")

            # Remove Markdown code blocks
            raw_response = re.sub(r"^```json\s*|\s*```$", "", raw_response, flags=re.MULTILINE).strip()

            # Parse JSON
            chunk_meta = json.loads(raw_response)

            # Merge results
            for k in metadata:
                if k == "skills":
                    metadata[k].extend(chunk_meta.get(k, []))
                elif k == "experience_years":
                    metadata[k] = max(metadata[k], chunk_meta.get(k, 0))
                else:
                    metadata[k] = metadata[k] or chunk_meta.get(k)

        except json.JSONDecodeError:
            logging.warning(f"[WARN] Failed to parse JSON for chunk {i}: {raw_response}")
        except Exception as e:
            logging.error(f"[ERROR] Chunk {i} processing error: {e}")

    # Post-process
    metadata["skills"] = list(set(metadata["skills"]))
    for field in ["name", "designation", "location"]:
        if not metadata[field]:
            metadata[field] = "Unknown"

    logging.debug(f"[DEBUG] Final extracted metadata: {metadata}")
    return metadata


def generate_jd(role, years, location, skills=None):
    prompt = f"""
    Write a Job Description in JSON format with keys:
    role, years_experience, location, skills, responsibilities.
    Role: {role}
    Years: {years}
    Location: {location}
    Skills: {skills or "Not specified"}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_content = resp.choices[0].message.content.strip()
    logging.debug("[DEBUG] LLM raw output: %s", raw_content)

    # Cleanup
    raw_content = re.sub(r"^```json\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()

    try:
        jd = json.loads(raw_content)
    except json.JSONDecodeError:
        logging.warning("[WARN] Failed to parse JSON, returning fallback JD")
        jd = {
            "role": role,
            "years_experience": years,
            "location": location,
            "skills": skills or [],
            "responsibilities": []
        }
    return jd


def score_resume_vs_jd(resume_meta, jd):
    jd_skills = set([s.lower() for s in jd["skills"]])
    candidate_skills = set([s.lower() for s in resume_meta.get("skills", [])])

    # Skill overlap
    overlap = jd_skills.intersection(candidate_skills)
    skills_score = (len(overlap) / len(jd_skills)) * 100 if jd_skills else 0

    # Experience score (smaller weight)
    exp_required = jd.get("years_experience", 0)
    exp_candidate = resume_meta.get("experience_years", 0)
    exp_score = min(exp_candidate / exp_required, 1.0) * 100 if exp_required else 0

    # Designation score (optional)
    designation = resume_meta.get("designation", "").lower()
    designation_score = 100 if jd["role"].lower() in designation else 0

    # Weighted final score
    final_score = (0.6 * skills_score) + (0.3 * exp_score) + (0.1 * designation_score)

    return {
        "skills_score": round(skills_score, 2),
        "experience_score": round(exp_score, 2),
        "designation_score": round(designation_score, 2),
        "final_score": round(final_score, 2),
    }


# -------- FastAPI Endpoints -------- #
@app.post("/upload_resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        try:
            file_path = f"./{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())

            text = read_resume(file_path)

            # Process metadata with CoT
            metadata = extract_metadata(text)

            # Embed paragraph-level chunks (better recall)
            embeddings = []
            for i, chunk in enumerate(chunk_text(text, 500)):
                emb = get_embedding(chunk)
                index.upsert(vectors=[{
                    "id": f"{metadata['name']}_{i}",
                    "values": emb,
                    "metadata": metadata
                }])
                embeddings.append(emb)

            uploaded.append({"filename": file.filename, "metadata": metadata, "chunks": len(embeddings)})

        except Exception as e:
            uploaded.append({"filename": file.filename, "error": str(e)})

    return {"uploaded_files": uploaded}


@app.post("/generate_jd")
async def create_jd(payload: dict = Body(...)):
    role = payload.get("role")
    years = int(payload.get("years", 0))
    location = payload.get("location")
    skills = payload.get("skills", "")
    skills_list = [s.strip() for s in skills.split(",")] if skills else []
    jd = generate_jd(role, years, location, skills_list)
    return jd


@app.post("/match_candidates")
async def match_candidates(jd: dict):
    jd_text = f"{jd['role']} in {jd['location']} requiring {jd['years_experience']} years experience. Skills: {', '.join(jd['skills'])}"
    emb = get_embedding(jd_text)

    results = index.query(vector=emb, top_k=5, include_metadata=True)
    scored = []
    for match in results["matches"]:
        resume_meta = match["metadata"]
        score = score_resume_vs_jd(resume_meta, jd)

        # ✅ Filter out candidates with no skill match
        if score["skills_score"] > 0:
            scored.append({"candidate": resume_meta, "score": score})

    return scored


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
