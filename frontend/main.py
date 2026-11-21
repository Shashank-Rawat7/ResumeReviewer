import sys
import os
import streamlit as st
import requests

# Get the root directory and add it to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


from frontend.display import (
    display_personal_info,
    display_skill_match,
    display_missing_skills,
    get_prioritized_view,
    get_recommendation_options,
    display_skill_gap_summary,
    display_prioritized_missing_skills,
    SKILL_ANALYSIS_UNAVAILABLE_MESSAGE,
    display_work_experience,
    display_projects,
    display_course_recommendations,
    build_session_options,
    display_ats_content
)

from frontend.credentials import register_user, login_user, fetch_session_data

# API Base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

# --- SESSION STATE ---
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "page" not in st.session_state:
    st.session_state.page = "login"  # Default page
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "session_data" not in st.session_state:
    st.session_state.session_data = {
        "resume_analysis": {},
        "skill_recommendation": {}
    }



# --- LOGIN PAGE ---
if st.session_state.page == "login":
    st.title("🔐 Resume Reviewer - Login or Register")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            login_user(username, password)

    with tab2:
        st.subheader("Register")
        new_username = st.text_input("New Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            register_user(new_username, new_email, new_password)

# --- SESSION SELECTOR PAGE ---
elif st.session_state.page == "session_select":
    st.title("📂 Select or Create a Session")
    headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}
    resp = requests.get(f"{API_BASE_URL}/my_sessions", headers=headers)

    if resp.status_code == 200:
        # Sessions are identified by their unique id; the label shows the name and the id.
        session_ids, session_labels = build_session_options(resp.json())
        create_choice = "➕ Create New Session"
        choice = st.selectbox(
            "Choose Session", session_ids + [create_choice],
            format_func=lambda option: session_labels.get(option, option)
        )

        if choice == create_choice:
            new_session_name = st.text_input("Enter New Session Name")
            if st.button("Create Session") and new_session_name:
                create_resp = requests.post(
                    f"{API_BASE_URL}/start_session",
                    params={"session_name": new_session_name},
                    headers=headers
                )
                if create_resp.status_code == 200:
                    st.session_state.session_id = create_resp.json()["session_id"]
                    fetch_session_data(st.session_state.session_id)
                    st.session_state.page = "main"
                    st.rerun()
                else:
                    st.error("❌ Failed to create session.")
        else:
            if st.button("Load Selected Session"):
                st.session_state.session_id = choice
                fetch_session_data(st.session_state.session_id)
                st.session_state.page = "main"
                st.rerun()
    else:
        st.error("❌ Unable to fetch sessions.")

# --- MAIN APP WITH SIDEBAR NAVIGATION ---
elif st.session_state.page == "main":
    st.sidebar.title("📁 Navigation")

    tab = st.sidebar.radio("Go to", [
        "Resume Analysis", "Skill Recommendations", "ATS Resume Bullets"
    ])
    if st.sidebar.button("🚪 Logout"):
        st.session_state.auth_token = None
        st.session_state.page = "login"
        st.rerun()

    headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}

        # --- RESUME ANALYSIS PAGE ---
    if tab == "Resume Analysis":
        st.title("📄 Resume and Job Description Analyzer")
        analysis = st.session_state.session_data.get("resume_analysis", {})

        # If no analysis data yet
        if not analysis:
            uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
            job_description = st.text_area("Paste Job Description")
            if st.button("Analyze Resume"):
                if uploaded_file and job_description:
                    # The backend stores the upload under a server-generated name that belongs to this user
                    upload_resp = requests.post(
                        f"{API_BASE_URL}/upload_resume",
                        files={"file": ("resume.pdf", uploaded_file.getvalue(), "application/pdf")},
                        headers=headers
                    )
                    if upload_resp.status_code != 200:
                        st.error("The resume could not be uploaded. Please try again.")
                        st.stop()

                    # Send request to backend for resume analysis
                    resp = requests.post(
                        f"{API_BASE_URL}/parse_resume",
                        data={
                            "resume_filename": upload_resp.json()["resume_filename"],
                            "job_description": job_description,
                            "session_id": st.session_state.session_id
                        },
                        headers=headers
                    )

                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state.session_data["resume_analysis"] = result
                        analysis = result
                        st.success("✅ Analysis complete!")
                        st.rerun()
                    elif resp.status_code >= 500:
                        st.error("The analysis could not be completed. Please try again.")
                    else:
                        st.error(f"Error: {resp.text}")
                else:
                    st.warning("Please upload a resume and enter a job description.")

        if analysis:
            # 📊 Beautiful Report Heading
            st.markdown("""
            Get a personalized analysis of your resume, highlighting strengths, skill gaps, and career highlights.
            """, unsafe_allow_html=True)

            summary = analysis.get("summary", {})

            # 👤 Personal Info & Education
            display_personal_info(summary.get("personal_info_and_education", {}))

            # Prioritised skill analysis state: "ok", "unavailable", or "legacy" (saved before it existed)
            skills_state, prioritized = get_prioritized_view(analysis)

            if skills_state == "legacy":
                # Intentional exception: a legacy session stores no evidence information, so it keeps the
                # original skill bar and "Missing Skills" wording instead of the evidence-based labels.
                # 🎯 Skill Match Progress Bars
                total_skills_len = len(analysis.get("job_skills", {}).get("technical_skills", []))
                missing_skills_len = len(analysis.get("missing_skills", {}).get("technical_skills", []))
                display_skill_match(missing_skills_len, total_skills_len)

                # ❗ Missing Skills as Chips
                display_missing_skills(analysis.get("missing_skills", {}))
            elif skills_state == "ok":
                display_skill_gap_summary(prioritized, analysis.get("job_skills", {}))
                display_prioritized_missing_skills(prioritized, analysis.get("job_skills", {}))
            else:
                # An empty legacy missing_skills here means "unavailable", not "nothing missing"
                st.warning(SKILL_ANALYSIS_UNAVAILABLE_MESSAGE)

            # 💼 Work Experience as Flip Cards
            display_work_experience(summary.get("work_experience", []))

            # 🚀 Projects as Carousel Flashcards
            display_projects(summary.get("key_projects", []))

            # 🌟 Soft Skills Cloud
            st.subheader("🤝 Soft Skills")
            soft_skills = summary.get("extracurricular_and_soft_skills", {}).get("soft_skills", [])
            if soft_skills:
                st.markdown(" ".join([f"<span style='background:#2196F3;color:white;padding:5px 10px;border-radius:20px;margin:5px;display:inline-block'>{skill}</span>" for skill in soft_skills]), unsafe_allow_html=True)
            else:
                st.markdown("_No soft skills found._")

            # 🔄 Button to Reset Analysis
            if st.button("🔄 New Analysis"):
                st.session_state.session_data["resume_analysis"] = {}
                st.rerun()


    # --- SKILL RECOMMENDATIONS PAGE ---
    elif tab == "Skill Recommendations":
        st.title("📘 Skill Recommendations")
        resume_analysis = st.session_state.session_data.get("resume_analysis", {})
        skills_state, prioritized = get_prioritized_view(resume_analysis)
        skipped_recommendations = 0
        if skills_state == "unavailable":
            st.warning(SKILL_ANALYSIS_UNAVAILABLE_MESSAGE)
            recommendation_options = {}
        elif skills_state == "ok":
            recommendation_options, skipped_recommendations = get_recommendation_options(prioritized)
        else:
            legacy_skills = resume_analysis.get("missing_skills", {}).get("technical_skills", [])
            recommendation_options = {legacy_skill: legacy_skill for legacy_skill in legacy_skills}
        if recommendation_options:
            # Current analyses offer only verified Not Demonstrated / Partially Demonstrated technical skills;
            # a legacy session keeps its original wording.
            selector_label = "Select a Not Demonstrated or Partially Demonstrated Skill" if skills_state == "ok" else "Select a Missing Skill"
            skill = st.selectbox(selector_label, list(recommendation_options), format_func=recommendation_options.get)
            if skipped_recommendations:
                st.caption(
                    f"{skipped_recommendations} flagged technical skill(s) are not offered because their "
                    "job-description quote could not be found."
                )
            if st.button("Get Recommendations"):
                resp = requests.get(f"{API_BASE_URL}/recommend_courses", params={"skill": skill}, headers=headers)
                try:
                    body = resp.json()
                except ValueError:
                    body = None
                display_course_recommendations(resp.status_code, body, skill)
        elif skills_state == "ok":
            if skipped_recommendations:
                st.info("The flagged technical skills could not be verified against the job description, so no recommendations are offered.")
            else:
                st.info("No eligible technical skills were identified for recommendations: none were Not Demonstrated or Partially Demonstrated with a verified job-description quote. This does not guarantee the resume meets the role.")
        elif skills_state == "legacy":
            st.info("No missing skills found. Analyze a resume first.")

    # --- ATS RESUME BULLETS PAGE ---
    elif tab == "ATS Resume Bullets":
        st.title("📑 ATS Optimized Resume Bullets")
        analysis = st.session_state.session_data.get("resume_analysis", {}).get("ats_content", {})
        display_ats_content(analysis)

    