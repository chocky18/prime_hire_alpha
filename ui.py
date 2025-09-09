import streamlit as st
import requests
import json
import ast
st.title("Resume Matcher (FastAPI + Streamlit)")

backend_url = "https://prime-hire-alpha.onrender.com"
# Upload resume
st.header("Upload Multiple Resumes")
resume_files = st.file_uploader(
    "Choose PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if resume_files and st.button("Upload Resumes"):
    files = []
    for f in resume_files:
        files.append(
            ("files", (f.name, f.getvalue(), f.type))
        )

    res = requests.post(f"{backend_url}/upload_resumes", files=files)

    if res.status_code == 200:
        st.success("All resumes processed!")
        st.json(res.json())
    else:
        st.error(f"Error uploading resumes: {res.status_code}")
        try:
            st.json(res.json())
        except:
            st.write(res.text)


# ---- Generate JD ----
st.header("Generate Job Description")
role = st.text_input("Role")
years = st.number_input("Years Experience", step=1, min_value=0)
location = st.text_input("Location")
skills = st.text_area("Skills (comma separated)")

if st.button("Generate JD"):
    data = {"role": role, "years": years, "location": location, "skills": skills}
    try:
        res = requests.post(f"{backend_url}/generate_jd", json=data)  # use json=, NOT data=
        res.raise_for_status()  # will raise an HTTPError for 4xx/5xx
        jd_json = res.json()  # safe now
        st.subheader("Generated Job Description (JSON)")
        st.json(jd_json)

        # Optional: formatted string for copy-paste
        st.subheader("JD String for Copy-Paste")
        st.text(str(jd_json))

    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error: {e} - {res.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except ValueError as e:  # JSON decode error
        st.error(f"Failed to decode JSON: {res.text}")


# ---- Match Candidates ----
st.header("Match Candidates to JD")
jd_input  = st.text_area("Paste JD JSON here")
if st.button("Match Candidates"):
    try:
        try:
            # Try parsing as Python dict first
            jd = ast.literal_eval(jd_input)
        except Exception:
            # Fallback to proper JSON
            jd = json.loads(jd_input)

        res = requests.post(f"{backend_url}/match_candidates", json=jd)
        res.raise_for_status()
        candidates = res.json()
        st.subheader("Candidate Matches")
        st.json(candidates)

    except Exception as e:
        st.error(f"Error: {e}")
