import sys
from pathlib import Path
import tempfile

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PROJECT PATH
# ============================================================

SRC_PATH = Path(__file__).parent / "src"

sys.path.append(str(SRC_PATH))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from resume import (
    extract_text_from_pdf,
    clean_resume_text
)

from jobs import get_all_jobs

from matching import calculate_job_matches

from rag_documents import create_job_documents

from rag import (
    create_document_embeddings,
    retrieve_documents
)

from rag_llm import generate_rag_explanation


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CareerAI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .upload-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .job-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
    }

    .job-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.3rem;
    }

    .company-location {
        color: #6b7280;
        margin-bottom: 1.2rem;
    }

    .skill-container {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .skill-badge {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        padding: 6px 11px;
        border-radius: 999px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .missing-badge {
        display: inline-block;
        background: #fef2f2;
        color: #b91c1c;
        padding: 6px 11px;
        border-radius: 999px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        padding: 12px;
        border-radius: 12px;
    }

    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # 💼 CareerAI

        **AI-Powered Job Matching**

        ---
        """
    )

    st.markdown("### 🔎 How it works")

    st.markdown(
        """
        **1. Upload Resume**

        Your PDF resume is processed and converted into text.

        **2. Analyze Skills**

        CareerAI identifies skills relevant to each job.

        **3. Calculate Similarity**

        TF-IDF and semantic embeddings compare your resume
        with job descriptions.

        **4. ML Ranking**

        A machine-learning model calculates a relevance score.

        **5. RAG Retrieval**

        Relevant job information is retrieved from PostgreSQL.

        **6. AI Explanation**

        An LLM explains the retrieved information using RAG context.
        """
    )

    st.divider()

    st.caption(
        "CareerAI — Intelligent Job Matching & Application Copilot"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💼 CareerAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Job Matching & Application Copilot'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# RESUME UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📄 Upload Your Resume</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-box">'
    '<b>Upload a PDF resume to start your analysis.</b>'
    '<br>'
    'CareerAI will compare your resume with available jobs, '
    'retrieve relevant job information using RAG, and generate '
    'AI-powered insights.'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Choose your resume",
    type=["pdf"],
    label_visibility="collapsed"
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file:

    st.success(
        f"✓ {uploaded_file.name} uploaded successfully"
    )

    analyze_button = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True
    )


    if analyze_button:

        with st.spinner(
            "CareerAI is analyzing your resume..."
        ):

            # ------------------------------------------------
            # SAVE UPLOADED RESUME
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                resume_path = temp_file.name


            # ------------------------------------------------
            # EXTRACT RESUME TEXT
            # ------------------------------------------------

            raw_resume_text = extract_text_from_pdf(
                resume_path
            )

            resume_text = clean_resume_text(
                raw_resume_text
            )


            # ------------------------------------------------
            # GET JOBS FROM POSTGRESQL
            # ------------------------------------------------

            jobs = get_all_jobs()


            # ------------------------------------------------
            # EXISTING JOB MATCHING
            # ------------------------------------------------

            matches = calculate_job_matches(
                resume_text,
                jobs
            )


            # ------------------------------------------------
            # CREATE MATCHING DATAFRAME
            # ------------------------------------------------

            rows = []

            for match in matches:

                job = match["job"]

                rows.append(
                    {
                        "job_title": job["title"],
                        "company": job["company"],
                        "location": job["location"],
                        "employment_type": job[
                            "employment_type"
                        ],
                        "experience_level": job[
                            "experience_level"
                        ],
                        "tfidf_score": match[
                            "tfidf_score"
                        ],
                        "semantic_score": match[
                            "semantic_score"
                        ],
                        "skill_match": match[
                            "skill_match_percentage"
                        ],
                        "matched_skills": match[
                            "matched_skills"
                        ],
                        "missing_skills": match[
                            "missing_skills"
                        ]
                    }
                )


            data = pd.DataFrame(rows)


            # ------------------------------------------------
            # ML RANKING MODEL
            # ------------------------------------------------

            model = joblib.load(
                "src/ranking_model.pkl"
            )


            X = data[
                [
                    "tfidf_score",
                    "semantic_score",
                    "skill_match"
                ]
            ]


            probabilities = model.predict_proba(X)


            data["ml_score"] = (
                probabilities[:, 1] * 100
            )


            # ------------------------------------------------
            # FINAL MATCH SCORE
            # ------------------------------------------------

            data["final_score"] = (
                0.40 * data["tfidf_score"]
                + 0.30 * (
                    data["skill_match"] / 100
                )
                + 0.30 * data["semantic_score"]
            ) * 100


            data = data.sort_values(
                "final_score",
                ascending=False
            ).reset_index(drop=True)


            # =================================================
            # RAG DOCUMENTS
            # =================================================

            rag_documents = create_job_documents()


            # ------------------------------------------------
            # CREATE DOCUMENT EMBEDDINGS
            # ------------------------------------------------

            rag_embeddings = create_document_embeddings(
                rag_documents
            )


            # ------------------------------------------------
            # CREATE RAG QUERY FROM RESUME
            # ------------------------------------------------

            rag_query = f"""
            Find jobs relevant to this candidate.

            Candidate Resume:
            {resume_text}
            """


            # ------------------------------------------------
            # RETRIEVE RELEVANT JOBS
            # ------------------------------------------------

            retrieved_documents = retrieve_documents(
                query=rag_query,
                documents=rag_documents,
                document_embeddings=rag_embeddings,
                top_k=3
            )


            # ------------------------------------------------
            # RAG + LLM EXPLANATION
            # ------------------------------------------------

            rag_explanation = generate_rag_explanation(
                query=rag_query,
                retrieved_documents=retrieved_documents
            )


        # =====================================================
        # TOP MATCHES
        # =====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🏆 Top Job Matches'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "CareerAI ranked the available jobs using "
            "resume-job similarity, skill alignment, and "
            "machine-learning relevance."
        )


        for index, row in data.head(3).iterrows():

            st.markdown(
                '<div class="job-card">',
                unsafe_allow_html=True
            )


            st.markdown(
                f'<div class="job-title">'
                f'#{index + 1} {row["job_title"]}'
                f'</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                f'<div class="company-location">'
                f'🏢 {row["company"]}'
                f' &nbsp; • &nbsp; '
                f'📍 {row["location"]}'
                f'</div>',
                unsafe_allow_html=True
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Match Score",
                    f"{row['final_score']:.2f}%"
                )


            with col2:

                st.metric(
                    "Skill Match",
                    f"{row['skill_match']:.2f}%"
                )


            with col3:

                st.metric(
                    "ML Relevance",
                    f"{row['ml_score']:.2f}%"
                )


            # ------------------------------------------------
            # MATCHED SKILLS
            # ------------------------------------------------

            st.markdown("**✅ Matched Skills**")

            matched_html = (
                '<div class="skill-container">'
            )


            for skill in row["matched_skills"]:

                matched_html += (
                    '<span class="skill-badge">'
                    f'{skill}'
                    '</span>'
                )


            matched_html += "</div>"


            st.markdown(
                matched_html,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # MISSING SKILLS
            # ------------------------------------------------

            st.markdown("**❌ Missing Skills**")

            missing_html = (
                '<div class="skill-container">'
            )


            for skill in row["missing_skills"]:

                missing_html += (
                    '<span class="missing-badge">'
                    f'{skill}'
                    '</span>'
                )


            missing_html += "</div>"


            st.markdown(
                missing_html,
                unsafe_allow_html=True
            )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # =====================================================
        # RAG + LLM SECTION
        # =====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🤖 RAG + AI Career Insights'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "CareerAI retrieves relevant job information "
            "from the PostgreSQL job database and provides "
            "an LLM-generated explanation based on that context."
        )


        with st.expander(
            "🔎 View Retrieved RAG Jobs"
        ):

            for rank, result in enumerate(
                retrieved_documents,
                start=1
            ):

                st.markdown(
                    f"### Rank {rank}"
                )

                st.write(
                    f"Semantic Similarity: "
                    f"{result['similarity']:.3f}"
                )

                st.text(
                    result["document"]
                )

                st.divider()


        with st.expander(
            "🧠 View RAG + LLM Explanation",
            expanded=True
        ):

            st.markdown(
                rag_explanation
            )


        # =====================================================
        # ALL JOB MATCHES
        # =====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📊 All Job Matches'
            '</div>',
            unsafe_allow_html=True
        )


        display_data = data[
            [
                "job_title",
                "company",
                "location",
                "skill_match",
                "ml_score",
                "final_score"
            ]
        ].copy()


        display_data.columns = [
            "Job",
            "Company",
            "Location",
            "Skill Match",
            "ML Relevance",
            "Final Match"
        ]


        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            column_config={

                "Skill Match":
                    st.column_config.ProgressColumn(
                        "Skill Match",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    ),

                "ML Relevance":
                    st.column_config.ProgressColumn(
                        "ML Relevance",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    ),

                "Final Match":
                    st.column_config.ProgressColumn(
                        "Final Match",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    )
            }
        )


        # =====================================================
        # TECHNICAL DETAILS
        # =====================================================

        with st.expander(
            "⚙️ View Technical Matching Details"
        ):

            st.markdown(
                """
                ### CareerAI Matching Pipeline

                **TF-IDF Similarity — 40%**

                Measures lexical similarity between the resume
                and job descriptions.

                **Skill Match — 30%**

                Measures how many required job skills are found
                in the resume.

                **Semantic Similarity — 30%**

                Uses sentence embeddings to compare the meaning
                of the resume and job descriptions.

                **ML Ranking**

                A prototype Random Forest model provides an
                additional relevance signal. It was trained
                using synthetic training data and is not a
                hiring probability.

                **RAG**

                Job information is converted into semantic
                embeddings. CareerAI retrieves the most relevant
                job documents from the PostgreSQL-backed dataset.

                **LLM**

                The LLM receives the retrieved RAG context and
                generates a grounded explanation using only
                the retrieved information.
                """
            )


else:

    st.info(
        "👆 Upload a PDF resume above to begin."
    )