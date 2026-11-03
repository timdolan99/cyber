import importlib
import json
import os
import re
from datetime import datetime
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# --- Orthology Branding Setup ---
LOGO_PATH = "ortho_logo.png"
ICON_PATH = "ortho_icon.png"

# --- Load Dynamic CompTIA Course Specification & Nodes ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(BASE_DIR, "comptia_secplus_spec.json")
NODES_PATH = os.path.join(BASE_DIR, "comptia_nodes.json")

if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {
        "course_title": "CompTIA Security+",
        "code": "SY0-701",
        "target_audience": "Adult Professional / Cyber Specialist",
        "socratic_mode": "Uncapped",
        "domains": {}
    }

if os.path.exists(NODES_PATH):
    with open(NODES_PATH, "r", encoding="utf-8") as f:
        COMPTIA_NODES = json.load(f)
else:
    COMPTIA_NODES = []

COURSE_TITLE = COURSE_SPEC.get("course_title", "CompTIA Security+")
EXAM_CODE = COURSE_SPEC.get("code", "SY0-701")

# --- Dynamic Page Configuration ---
st.set_page_config(
    page_title=f"{COURSE_TITLE} ({EXAM_CODE}) - Orthology Socratic Engine",
    page_icon=ICON_PATH if os.path.exists(ICON_PATH) else None,
    layout="wide",
)

if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.sidebar.columns([1, 6, 1])
    with col2:
        st.image(LOGO_PATH, use_container_width=True)

# Ensure API Key is set BEFORE loading CAG FSM
raw_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if isinstance(raw_key, dict):
    raw_key = raw_key.get("GOOGLE_API_KEY") or raw_key.get("GEMINI_API_KEY")
if raw_key:
    os.environ["GOOGLE_API_KEY"] = str(raw_key).strip()
    os.environ["GEMINI_API_KEY"] = str(raw_key).strip()

import socratic_cag_fsm as fsm
importlib.reload(fsm)

# --- Warm-up Server-Side Context Cache once on Streamlit Startup ---
@st.cache_resource
def init_cag_cache_handle():
    return fsm.get_or_create_cag_cache()

cag_cache = init_cag_cache_handle()

from socratic_cag_fsm import (
    stream_socratic_response,
    verify_topic_alignment,
    stream_freestyle_socratic_response,
    get_suggested_prompts,
    generate_refactor_challenge,
    grade_refactored_response,
    generate_pbq_scenario,
    grade_pbq_response,
    generate_socratic_synthesis_prompt,
    evaluate_socratic_synthesis,
    load_questions_from_syllabus_and_nodes
)

# --- COLOR THEME CONFIGURATION ---
THEME_PRIMARY = "#0e7490"        # Deep Cyan Primary Accent
THEME_GRADIENT_START = "#0f172a"  # Slate Dark Navy Header Start
THEME_GRADIENT_END = "#1e293b"    # Dark Slate Header End
THEME_BANNER_BG = "#f0fdf4"       # Mint Green Purpose Banner
THEME_BANNER_BORDER = "#bbf7d0"   # Soft Green Border
THEME_BANNER_TEXT = "#166534"     # Dark Green Banner Text

# --- Database & Zero-PII Telemetry Logging Setup ---
Base = declarative_base()

class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    subject = Column(Text, nullable=False)
    level = Column(Text, nullable=False)
    unit = Column(Text, nullable=False)
    subtopic = Column(Text, nullable=False)
    app_mode = Column(Text, nullable=False)
    score_pct = Column(Float, nullable=False)
    keywords_used = Column(Text, nullable=True)
    keywords_missed = Column(Text, nullable=True)
    misconception_flag = Column(Integer, default=0)

def log_session_to_supabase(unit: str, subtopic: str, app_mode: str, score_pct: float, keywords_used: list = None, keywords_missed: list = None, misconception_flag: int = 0):
    """Logs session metrics statelessly to Supabase activity_logs table, bypassing quietly if unconfigured."""
    try:
        if "postgres" in st.secrets and "url" in st.secrets["postgres"] and st.secrets["postgres"]["url"]:
            db_url = st.secrets["postgres"]["url"]
            engine = create_engine(db_url, pool_pre_ping=True)
            Session = sessionmaker(bind=engine)
            session = Session()

            kw_used_str = ", ".join([str(k) for k in keywords_used if k]) if keywords_used else None
            kw_missed_str = ", ".join([str(k) for k in keywords_missed if k]) if keywords_missed else None

            log_entry = ActivityLog(
                subject=f"{COURSE_TITLE} ({EXAM_CODE})",
                level="Adult Professional",
                unit=unit or "General Domain",
                subtopic=subtopic or "General Objective",
                app_mode=app_mode,
                score_pct=round(float(score_pct), 1),
                keywords_used=kw_used_str,
                keywords_missed=kw_missed_str,
                misconception_flag=int(misconception_flag)
            )

            session.add(log_entry)
            session.commit()
            session.close()
    except Exception:
        pass

# --- Formatting Helpers ---
def extract_clean_text(response) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return extract_clean_text(response.content)
    return str(response)

def md_to_html(text: str) -> str:
    text = re.sub(r'^####\s+(.*$)', r'<h4 style="margin: 4px 0 1px 0; font-size: 1.05em; color: inherit;">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*$)', r'<h3 style="margin: 6px 0 2px 0; font-size: 1.1em; color: inherit;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*$)', r'<h2 style="margin: 8px 0 2px 0; font-size: 1.2em; color: inherit;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'^\s*[-*]\s+(.*$)', r'<div style="margin: 1px 0;">• \1</div>', text, flags=re.MULTILINE)
    text = re.sub(r'(</(div|h2|h3|h4)>)\s*\n+', r'\1', text)
    text = re.sub(r'\n{2,}', '<br>', text)
    text = text.replace('\n', '<br>')
    return re.sub(r'(<br\s*/?>\s*)+', '<br>', text)

# --- CSS Styling ---
st.markdown(f"""
    <style>
    .stApp {{ background: #f8fafc; }}
    
    div[data-testid="stSidebar"], 
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] > div {{
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }}
    
    div[data-testid="stCodeBlock"] pre, 
    div[data-testid="stCodeBlock"] code,
    code {{
        white-space: pre-wrap !important;
        word-break: break-word !important;
        overflow-x: auto !important;
    }}

    div[data-testid="stProgress"] > div > div > div {{ background-color: {THEME_PRIMARY} !important; }}
    h1, h2, h3 {{ color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 700; }}
    
    .chat-header {{ 
        background: linear-gradient(135deg, {THEME_GRADIENT_START} 0%, {THEME_GRADIENT_END} 100%); 
        color: white; padding: 22px; font-weight: 700; text-align: center; 
        font-size: 1.3em; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.15);
        margin-bottom: 20px;
    }}
    .purpose-banner {{
        background: {THEME_BANNER_BG}; border: 1px solid {THEME_BANNER_BORDER}; padding: 14px 18px;
        border-radius: 12px; color: {THEME_BANNER_TEXT}; font-size: 0.95em; margin-bottom: 20px;
        line-height: 1.4;
    }}
    .tutor-msg {{ 
        background-color: #ffffff; color: #1e293b; padding: 14px 18px; 
        border-radius: 18px 18px 18px 4px; margin-bottom: 12px; max-width: 82%; 
        line-height: 1.35; border: 1px solid #e2e8f0;
    }}
    .student-msg {{ 
        background: linear-gradient(135deg, {THEME_GRADIENT_START} 0%, {THEME_GRADIENT_END} 100%); 
        color: white; padding: 14px 18px; border-radius: 18px 18px 4px 18px; 
        margin-bottom: 12px; max-width: 82%; margin-left: auto; line-height: 1.35;
    }}
    .stButton > button {{ border-radius: 12px !important; font-weight: 600 !important; margin-top: 4px !important; margin-bottom: 4px !important; }}
    button[kind="primary"] {{
        background-color: {THEME_PRIMARY} !important;
        border-color: {THEME_PRIMARY} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "active_domain" not in st.session_state:
    st.session_state.active_domain = None
if "active_node_id" not in st.session_state:
    st.session_state.active_node_id = None
if "active_subtopic" not in st.session_state:
    st.session_state.active_subtopic = None
if "is_freestyle" not in st.session_state:
    st.session_state.is_freestyle = False
if "freestyle_prompt" not in st.session_state:
    st.session_state.freestyle_prompt = ""
if "freestyle_focus" not in st.session_state:
    st.session_state.freestyle_focus = "Technical Architecture & Implementation"
if "app_mode" not in st.session_state:
    st.session_state.app_mode = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "refactor_data" not in st.session_state:
    st.session_state.refactor_data = None
if "refactor_results" not in st.session_state:
    st.session_state.refactor_results = None
if "refactor_retry_mode" not in st.session_state:
    st.session_state.refactor_retry_mode = False
if "last_submitted_refactor" not in st.session_state:
    st.session_state.last_submitted_refactor = ""
if "pbq_data" not in st.session_state:
    st.session_state.pbq_data = None
if "pbq_results" not in st.session_state:
    st.session_state.pbq_results = None
if "socratic_dialogue_complete" not in st.session_state:
    st.session_state.socratic_dialogue_complete = False
if "socratic_synthesis_q" not in st.session_state:
    st.session_state.socratic_synthesis_q = None
if "socratic_synthesis_results" not in st.session_state:
    st.session_state.socratic_synthesis_results = None
if "exam_question_pool" not in st.session_state:
    st.session_state.exam_question_pool = []
if "exam_current_index" not in st.session_state:
    st.session_state.exam_current_index = 0
if "exam_user_answers" not in st.session_state:
    st.session_state.exam_user_answers = {}
if "exam_submitted" not in st.session_state:
    st.session_state.exam_submitted = False

def reset_session():
    st.session_state.active_domain = None
    st.session_state.active_node_id = None
    st.session_state.active_subtopic = None
    st.session_state.is_freestyle = False
    st.session_state.freestyle_prompt = ""
    st.session_state.freestyle_focus = "Technical Architecture & Implementation"
    st.session_state.app_mode = None
    st.session_state.messages = []
    st.session_state.refactor_data = None
    st.session_state.refactor_results = None
    st.session_state.refactor_retry_mode = False
    st.session_state.last_submitted_refactor = ""
    st.session_state.pbq_data = None
    st.session_state.pbq_results = None
    st.session_state.socratic_dialogue_complete = False
    st.session_state.socratic_synthesis_q = None
    st.session_state.socratic_synthesis_results = None
    st.session_state.exam_question_pool = []
    st.session_state.exam_current_index = 0
    st.session_state.exam_user_answers = {}
    st.session_state.exam_submitted = False
    st.rerun()

# --- Sidebar Exam Mode Trigger ---
with st.sidebar:
    st.write("---")
    if st.button("Simulate CompTIA Exam Mode", use_container_width=True):
        st.session_state.app_mode = "exam_mode"
        st.session_state.exam_question_pool = []
        st.session_state.exam_current_index = 0
        st.session_state.exam_user_answers = {}
        st.session_state.exam_submitted = False
        st.rerun()

# --- Dynamic Screen Router ---

if st.session_state.active_subtopic is None and st.session_state.app_mode not in ["freestyle_input", "exam_mode"]:
    st.markdown(f'<div class="chat-header">Orthology Socratic Engine — {COURSE_TITLE} ({EXAM_CODE})</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="purpose-banner">
            <b>Uncapped Socratic Assessment:</b> This engine tests technical precision, mandatory acronym usage, 
            and architectural reasoning for CompTIA Security+. Select a domain objective or launch a freestyle scenario to begin.
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Select Objective Target")
    
    domains_dict = COURSE_SPEC.get("domains", {})
    if domains_dict:
        sel_domain = st.selectbox("Step 1: Select Security+ Domain:", options=list(domains_dict.keys()))
        sub_list = domains_dict.get(sel_domain, [])
        sel_subtopic = st.selectbox("Step 2: Select Sub-Objective:", options=sub_list)
        
        target_node_id = "1.2_fundamental_security_concepts"
        for node in COMPTIA_NODES:
            if node.get("domain") == sel_domain:
                sub_obj_text = node.get("sub_objective", "")
                node_code = node.get("node_id", "").split("_")[0]
                if sel_subtopic.startswith(node_code) or sel_subtopic in sub_obj_text:
                    target_node_id = node.get("node_id")
                    break
    else:
        sel_domain = "1.0 General Security Concepts"
        sel_subtopic = "1.2 Summarize fundamental security concepts"
        target_node_id = "1.2_fundamental_security_concepts"

    st.write("")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Start Guided Socratic Drill", type="primary", use_container_width=True):
            st.session_state.app_mode = "socratic_drill"
            st.session_state.is_freestyle = False
            st.session_state.active_domain = sel_domain
            st.session_state.active_subtopic = sel_subtopic
            st.session_state.active_node_id = target_node_id
            st.rerun()

    with col_btn2:
        if st.button("Start Freestyle Scenario Practice", type="primary", use_container_width=True):
            st.session_state.app_mode = "freestyle_input"
            st.rerun()

    st.write("---")
    st.subheader("Specialized Practice Modes")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if st.button("Refactor Flawed Response", use_container_width=True):
            st.session_state.app_mode = "response_refactor"
            st.session_state.active_domain = sel_domain
            st.session_state.active_subtopic = sel_subtopic
            st.session_state.active_node_id = target_node_id
            st.session_state.refactor_data = generate_refactor_challenge(target_node_id)
            st.rerun()

    with col_m2:
        if st.button("Text-Based PBQ Log Analysis", use_container_width=True):
            st.session_state.app_mode = "pbq_tester"
            st.session_state.active_domain = sel_domain
            st.session_state.active_subtopic = sel_subtopic
            st.session_state.active_node_id = target_node_id
            st.session_state.pbq_data = generate_pbq_scenario(target_node_id)
            st.rerun()

# --- Mock Exam Mode View ---
elif st.session_state.app_mode == "exam_mode":
    st.markdown(f'<div class="chat-header">CompTIA Security+ Mock Exam Simulation (45 Items)</div>', unsafe_allow_html=True)
    
    timed_mode = st.sidebar.checkbox("Enable 45-Minute Exam Timer", value=False)
    if timed_mode:
        st.sidebar.info("⏱️ Simulation mode active: 45-minute pacing monitored.")

    if not st.session_state.exam_question_pool:
        with st.spinner("Initializing 45-question CompTIA Security+ examination pool (Front-Loaded Open PBQs, Acronyms & MCQs)..."):
            st.session_state.exam_question_pool = load_questions_from_syllabus_and_nodes()
        st.session_state.exam_current_index = 0
        st.session_state.exam_user_answers = {}
        st.session_state.exam_submitted = False

    pool = st.session_state.exam_question_pool
    idx = st.session_state.exam_current_index

    if pool and not st.session_state.exam_submitted:
        q = pool[idx]
        is_pbq = q.get("type") == "pbq"
        badge = "🔥 Performance-Based Question (Open-Ended Log Analysis)" if is_pbq else f"Domain: {q['domain']}"
        st.progress((idx + 1) / len(pool), text=f"Question {idx + 1} of {len(pool)} | {badge}")
        
        st.subheader(f"Objective: {q['sub_objective']}")
        st.markdown(q["question"])
        
        if is_pbq:
            if q.get("log_snippet"):
                st.code(q["log_snippet"], language="text")
            
            current_ans = st.session_state.exam_user_answers.get(idx, "")
            user_analysis = st.text_area(
                "Provide your written troubleshooting analysis (Attack Vector, Protocol, Technical Remediation):",
                value=current_ans,
                height=180,
                key=f"exam_pbq_text_{idx}"
            )
            st.session_state.exam_user_answers[idx] = user_analysis
        else:
            current_ans = st.session_state.exam_user_answers.get(idx, None)
            selected_option = st.radio(
                "Select your technical evaluation response:",
                options=q["options"],
                index=q["options"].index(current_ans) if current_ans in q["options"] else 0,
                key=f"exam_radio_{idx}"
            )
            st.session_state.exam_user_answers[idx] = selected_option

        st.write("---")
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        with col_nav1:
            if idx > 0 and st.button("⬅️ Previous", use_container_width=True):
                st.session_state.exam_current_index -= 1
                st.rerun()
        with col_nav2:
            if idx < len(pool) - 1 and st.button("Next ➡️", type="primary", use_container_width=True):
                st.session_state.exam_current_index += 1
                st.rerun()
        with col_nav3:
            if idx == len(pool) - 1 and st.button("🏁 Finish & Score Exam", type="primary", use_container_width=True):
                st.session_state.exam_submitted = True
                st.rerun()

    elif st.session_state.exam_submitted:
        total_q = len(pool)
        correct_count = 0
        domain_stats = {}
        pbq_scores_sum = 0
        pbq_total = 0
        missed_domains = set()
        missed_questions_text = []
        
        with st.spinner("Grading examination items and analyzing written PBQ log troubleshooting..."):
            for i, q in enumerate(pool):
                user_ans = st.session_state.exam_user_answers.get(i, "")
                is_pbq = q.get("type") == "pbq"
                
                raw_dom = q["domain"]
                if not raw_dom or "Syllabus" in raw_dom or ".pdf" in raw_dom:
                    clean_dom = "1.0 General Security Concepts"
                else:
                    clean_dom = raw_dom
                
                if clean_dom not in domain_stats:
                    domain_stats[clean_dom] = {"correct": 0, "total": 0}
                domain_stats[clean_dom]["total"] += 1

                if is_pbq:
                    pbq_total += 1
                    pbq_grade = grade_pbq_response(
                        node_id="pbq_eval",
                        log_snippet=q.get("log_snippet", q["question"]),
                        student_analysis=user_ans
                    )
                    pbq_score = pbq_grade.get("score", 0)
                    pbq_scores_sum += pbq_score
                    if pbq_score >= 70:
                        correct_count += 1
                        domain_stats[clean_dom]["correct"] += 1
                    else:
                        missed_domains.add(clean_dom)
                        missed_questions_text.append(q["question"] + " " + q.get("sub_objective", ""))
                else:
                    correct_choice = q["options"][q["correct_index"]]
                    if user_ans == correct_choice:
                        correct_count += 1
                        domain_stats[clean_dom]["correct"] += 1
                    else:
                        missed_domains.add(clean_dom)
                        missed_questions_text.append(q["question"] + " " + " ".join(q.get("options", [])))

        percentage = (correct_count / total_q) if total_q > 0 else 0
        scaled_score = int(100 + (percentage * 800))
        passed = scaled_score >= 750

        st.subheader("Exam Simulation Results")
        if passed:
            st.success(f"🎉 **PASS! Scaled Score: {scaled_score} / 900** (Passing threshold is 750)")
        else:
            st.error(f"⚠️ **NON-PASS. Scaled Score: {scaled_score} / 900** (Passing threshold is 750)")

        st.markdown(f"**Overall Correct / Verified:** {correct_count} / {total_q} items ({percentage * 100:.1f}%)")
        
        if pbq_total > 0:
            avg_pbq = pbq_scores_sum / pbq_total
            st.info(f"🔥 **Front-Loaded PBQ Log Analysis Performance:** Average Score: {avg_pbq:.1f}/100. Evaluated via rigorous written technical troubleshooting analysis.")

        st.markdown("### Granular Domain Performance Breakdown:")
        for dom, stats in domain_stats.items():
            dom_pct = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            st.markdown(f"- **{dom}:** {stats['correct']}/{stats['total']} correct ({dom_pct:.0f}%)")

        st.markdown("### 🧠 Granular Examiner Misconception & Literacy Analysis:")
        
        # Filter strictly for domains scoring below 75%
        true_missed_domains = [dom for dom, stats in domain_stats.items() if (stats["correct"] / stats["total"]) < 0.75]

        if true_missed_domains:
            missed_list_str = ", ".join(true_missed_domains)
            
            # Extract specific acronyms associated with missed items
            flagged_acronyms = []
            combined_missed_blob = " ".join(missed_questions_text)
            potential_acros = re.findall(r'\b[A-Z]{2,6}\b', combined_missed_blob)
            
            # Exclude common stop words or headers
            stopwords = {"TCP", "SYN", "GET", "HTTP", "PDF", "API"}
            unique_acron_gaps = [a for a in list(set(potential_acros)) if a not in stopwords][:6]
            acronym_str = ", ".join(unique_acron_gaps) if unique_acron_gaps else "SIEM, PKI, LDAP, RPO/RTO"

            st.warning(
                f"**Targeted Conceptual Gaps Identified:** Your performance dropped below threshold in: **{missed_list_str}**.\n\n"
                f"• **Key Acronyms & Terms to Review:** `{acronyms_str}`\n" if 'acronyms_str' in locals() else f"• **Key Acronyms & Terms to Review:** `{acronym_str}`\n" +
                "• **Technical Recommendations:** Review specific protocol headers, secure deployment boundaries, "
                "and log anomaly signatures for these specific weak domains."
            )
        else:
            st.success("Flawless technical execution! Zero conceptual misconceptions or domain gaps detected.")

        log_session_to_supabase(
            unit="Mock Exam Simulation",
            subtopic="45-Item Diagnostic (Open PBQs, Acronyms, MCQs)",
            app_mode="exam_mode",
            score_pct=percentage * 100.0,
            misconception_flag=0 if passed else 1
        )

        if st.button("🔄 Start New Exam Session", use_container_width=True):
            st.session_state.exam_question_pool = []
            st.session_state.exam_submitted = False
            reset_session()

    if st.button("Exit to Main Menu", use_container_width=True):
        reset_session()

# --- Freestyle Input View ---
elif st.session_state.app_mode == "freestyle_input":
    st.markdown(f'<div class="chat-header">Freestyle Security+ Scenario Practice</div>', unsafe_allow_html=True)
    st.subheader("Formulate Any Custom Enterprise Cyber Scenario")

    suggested = get_suggested_prompts("freestyle")
    with st.expander("Suggested Prompt & Exemplar Prompting Style", expanded=True):
        st.info(f"**Instruction:** {suggested.get('instruction', '')}")
        st.code(suggested.get("suggested_prompt", ""), language="text")
        st.markdown("**Prompting Tips:**")
        for tip in suggested.get("tips", []):
            st.markdown(f"• {tip}")

    custom_input = st.text_area(
        "Enter your custom scenario or investigation query:",
        value=suggested.get("suggested_prompt", ""),
        height=140,
        key="freestyle_custom_input"
    )

    freestyle_focus = st.radio(
        "Select Pedagogical Focus:",
        ["Technical Architecture & Implementation", "Exam Terminology & Acronym Precision"],
        horizontal=True,
        key="freestyle_focus_selector"
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if st.button("Launch Freestyle Dialogue", type="primary", use_container_width=True):
            if custom_input.strip():
                with st.spinner("Verifying domain scope alignment..."):
                    alignment = verify_topic_alignment(custom_input.strip())

                if alignment.get("is_aligned", True):
                    st.session_state.app_mode = "socratic_drill"
                    st.session_state.is_freestyle = True
                    st.session_state.active_domain = "Freestyle Practice"
                    st.session_state.active_subtopic = custom_input.strip()[:60] + "..."
                    st.session_state.freestyle_prompt = custom_input.strip()
                    st.session_state.freestyle_focus = freestyle_focus
                    st.rerun()
                else:
                    st.error(f"**Domain Alignment Guardrail:** {alignment.get('reasoning', 'Topic out of Security+ scope.')}")
            else:
                st.warning("Please enter a scenario prompt first.")

    with col_f2:
        if st.button("Cancel / Back to Objective Selection", use_container_width=True):
            reset_session()

# --- Socratic Drill View ---
elif st.session_state.app_mode == "socratic_drill":
    st.markdown(f'<div class="chat-header">Orthology Socratic Coach</div>', unsafe_allow_html=True)
    
    student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")

    with st.sidebar:
        st.subheader("Active Target")
        if st.session_state.is_freestyle:
            st.info(f"**Mode:** Freestyle Scenario\n\n**Focus:** {st.session_state.freestyle_focus}\n\n**Prompt:** {st.session_state.active_subtopic}")
        else:
            st.info(f"**Domain:** {st.session_state.active_domain}\n\n**Sub-Objective:** {st.session_state.active_subtopic}")
        
        st.metric(label="Exchanges Count", value=f"{student_turns} (Uncapped)")
        
        if student_turns >= 1 and not st.session_state.socratic_dialogue_complete:
            if st.button("Synthesize & Exit Session", type="primary", use_container_width=True):
                st.session_state.socratic_dialogue_complete = True
                st.rerun()

        st.write("---")
        if st.button("Change Objective / Mode", use_container_width=True):
            reset_session()

    suggested = get_suggested_prompts("socratic_drill", st.session_state.active_node_id or "")
    with st.expander("Suggested Prompt & Exemplar Writing Style", expanded=False):
        st.info(suggested.get("instruction", ""))
        st.code(suggested.get("suggested_prompt", ""), language="text")
        st.markdown("**Prompting Guidance:**")
        for tip in suggested.get("tips", []):
            st.markdown(f"• {tip}")

    if len(st.session_state.messages) == 0:
        if st.session_state.is_freestyle:
            initial_greeting = f"Welcome! Let's examine your custom scenario: **'{st.session_state.freestyle_prompt}'**. To begin, what exact security controls or protocols are active in this environment?"
        else:
            initial_greeting = f"Welcome! We are evaluating sub-objective **{st.session_state.active_subtopic}**. To start, how do you implement these controls to guarantee security across the enterprise?"
        st.session_state.messages.append({"role": "tutor", "content": initial_greeting, "style": "tutor-msg"})

    for msg in st.session_state.messages:
        html_content = md_to_html(msg["content"])
        if msg["role"] == "tutor":
            div_class = msg.get("style", "tutor-msg")
            st.markdown(f'<div class="{div_class}"><b>Orthology Engine</b><br><br>{html_content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="student-msg"><b>Candidate</b><br><br>{html_content}</div>', unsafe_allow_html=True)

    is_disabled = st.session_state.socratic_dialogue_complete
    
    if user_input := st.chat_input("Type your response here using precise Security+ terms...", disabled=is_disabled):
        st.session_state.messages.append({"role": "student", "content": user_input})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "student":
        with st.status("Analyzing your answer against Security+ objective rubrics...", expanded=True) as status:
            st.write("• Checking required acronyms & technical terms...")
            st.write("• Evaluating architectural control mechanisms...")
            
            full_stream = ""
            if st.session_state.is_freestyle:
                generator = stream_freestyle_socratic_response(
                    messages=st.session_state.messages,
                    custom_prompt=st.session_state.freestyle_prompt,
                    focus_mode=st.session_state.freestyle_focus
                )
            else:
                generator = stream_socratic_response(
                    messages=st.session_state.messages,
                    sub_topic=st.session_state.active_subtopic,
                    node_id=st.session_state.active_node_id or ""
                )
            
            for chunk in generator:
                full_stream += extract_clean_text(chunk)
                
            status.update(label="Analysis complete!", state="complete", expanded=False)

        st.session_state.messages.append({"role": "tutor", "content": full_stream, "style": "tutor-msg"})
        st.rerun()

    if st.session_state.socratic_dialogue_complete:
        st.write("---")
        st.subheader("Final Mastery Synthesis (6-Point Check)")

        if st.session_state.socratic_synthesis_q is None:
            with st.spinner("Generating synthesis prompt based on session transcript..."):
                st.session_state.socratic_synthesis_q = generate_socratic_synthesis_prompt(
                    sub_topic=st.session_state.active_subtopic,
                    chat_history=st.session_state.messages
                )

        s_data = st.session_state.socratic_synthesis_q

        if st.session_state.socratic_synthesis_results is None:
            st.info(f"**Task Prompt:** {s_data['question']}")
            user_synth_ans = st.text_area("Write your technical synthesis here:", height=180, key="synth_ans_input")
            
            if st.button("Submit Synthesis for Evaluation", type="primary", use_container_width=True):
                if user_synth_ans.strip():
                    with st.spinner("Evaluating terminology precision..."):
                        eval_res = evaluate_socratic_synthesis(
                            question=s_data['question'],
                            mark_scheme=s_data['mark_scheme'],
                            student_answer=user_synth_ans.strip(),
                            chat_history=st.session_state.messages
                        )
                        st.session_state.socratic_synthesis_results = eval_res
                        
                        score_pct = (eval_res.get("score", 0) / 6.0) * 100.0
                        log_session_to_supabase(
                            unit=st.session_state.active_domain,
                            subtopic=st.session_state.active_subtopic,
                            app_mode="socratic_drill",
                            score_pct=score_pct,
                            keywords_used=eval_res.get("keywords_used", []),
                            keywords_missed=eval_res.get("keywords_missed", []),
                            misconception_flag=1 if score_pct < 50.0 else 0
                        )
                        st.rerun()
        else:
            res = st.session_state.socratic_synthesis_results
            st.success(f"**Synthesis Score: {res.get('score', 0)} / 6**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Points Awarded")
                for pt in res.get("awarded_points", []):
                    st.markdown(f"• {pt}")
            with col_b:
                st.markdown("### Missed Points")
                for pt in res.get("missed_points", []):
                    st.markdown(f"• {pt}")
            st.info(f"**Examiner Note:** {res.get('feedback', '')}")
            if st.button("Start New Session", use_container_width=True):
                reset_session()

# --- Response Refactor View ---
elif st.session_state.app_mode == "response_refactor":
    st.markdown(f'<div class="chat-header">Response Refactor Challenge</div>', unsafe_allow_html=True)
    
    ref_data = st.session_state.get("refactor_data", {})
    suggested = get_suggested_prompts("response_refactor", st.session_state.active_node_id or "")

    with st.expander("View Refactoring Exemplar Style", expanded=True):
        st.info(suggested.get("instruction", ""))
        st.code(suggested.get("suggested_prompt", ""), language="text")

    st.subheader("Flawed Initial Draft")
    st.warning(f'"{ref_data.get("flawed_text", "")}"')

    if st.session_state.refactor_results is None:
        student_refactor = st.text_area(
            "Rewrite this response using precise Security+ terminology:", 
            height=180, 
            key="refactor_input_box"
        )

        if st.button("Submit Refactored Response", type="primary", use_container_width=True):
            if student_refactor.strip():
                st.session_state["last_submitted_refactor"] = student_refactor.strip()
                with st.spinner("Evaluating refactored prose..."):
                    results = grade_refactored_response(
                        node_id=st.session_state.active_node_id or "",
                        flawed_text=ref_data.get("flawed_text", ""),
                        student_refactor=student_refactor.strip()
                    )
                    st.session_state.refactor_results = results
                    score = results.get("score", 0)
                    log_session_to_supabase(
                        unit=st.session_state.active_domain,
                        subtopic=st.session_state.active_subtopic,
                        app_mode="response_refactor",
                        score_pct=score,
                        keywords_used=results.get("matched_acronyms", []),
                        keywords_missed=results.get("missed_terms", []),
                        misconception_flag=0 if results.get("is_correct") else 1
                    )
                    st.rerun()
            else:
                st.warning("Please type a refactored answer before submitting.")

    else:
        res = st.session_state.refactor_results
        current_score = res.get('score', 0)

        with st.expander("View Task Prompt & Your Submission", expanded=True):
            st.markdown(f"**Flawed Draft:** *\"{ref_data.get('flawed_text', '')}\"*")
            st.markdown(f"**Your Answer:**\n\n> {st.session_state.get('last_submitted_refactor', '')}")

        st.success(f"**Refactor Graded! Score: {current_score} / 100**")
        st.info(f"**Examiner Feedback:** {res.get('feedback', '')}")
        st.write("")

        if current_score < 100:
            if not st.session_state.get("refactor_retry_mode", False):
                if st.button("Upgrade Response (Retry Challenge)", type="primary", use_container_width=True):
                    st.session_state.refactor_retry_mode = True
                    st.rerun()
            else:
                st.markdown("---")
                st.subheader("Upgrade Your Refactored Response")
                st.caption("Refine your answer by incorporating the missing terms and correcting any remaining inaccuracies highlighted above:")
                
                retry_text = st.text_area(
                    "Revised Refactored Answer:", 
                    value=st.session_state.get("last_submitted_refactor", ""), 
                    height=180, 
                    key="refactor_retry_input"
                )
                
                if st.button("Submit Upgraded Response for Re-Evaluation", type="primary", use_container_width=True):
                    if retry_text.strip():
                        st.session_state["last_submitted_refactor"] = retry_text.strip()
                        with st.spinner("Re-evaluating upgraded response..."):
                            eval_res = grade_refactored_response(
                                node_id=st.session_state.active_node_id or "",
                                flawed_text=ref_data.get("flawed_text", ""),
                                student_refactor=retry_text.strip()
                            )
                            st.session_state.refactor_results = eval_res
                            st.session_state.refactor_retry_mode = False
                            
                            new_score = eval_res.get("score", 0)
                            log_session_to_supabase(
                                unit=st.session_state.active_domain,
                                subtopic=st.session_state.active_subtopic,
                                app_mode="response_refactor_retry",
                                score_pct=new_score,
                                keywords_used=eval_res.get("matched_acronyms", []),
                                keywords_missed=eval_res.get("missed_terms", []),
                                misconception_flag=0 if eval_res.get("is_correct") else 1
                            )
                            st.rerun()

        st.write("")
        if st.button("Select Another Objective", use_container_width=True):
            reset_session()

# --- PBQ Log Tester View ---
elif st.session_state.app_mode == "pbq_tester":
    st.markdown(f'<div class="chat-header">Text-Based PBQ Log Analysis</div>', unsafe_allow_html=True)
    
    pbq_data = st.session_state.get("pbq_data", {})
    suggested = get_suggested_prompts("pbq_tester", st.session_state.active_node_id or "")

    with st.expander("View PBQ Analysis Structure", expanded=True):
        st.info(suggested.get("instruction", ""))
        st.code(suggested.get("suggested_prompt", ""), language="text")

    st.subheader("Log Snippet")
    st.code(pbq_data.get("log_snippet", ""), language="text")

    student_analysis = st.text_area("Provide your 3-part analysis (Attack Vector, OSI/Protocol, Mitigation):", height=200)

    if st.button("Submit PBQ Analysis", type="primary", use_container_width=True):
        if student_analysis.strip():
            with st.spinner("Grading log analysis against Security+ rubrics..."):
                results = grade_pbq_response(
                    node_id=st.session_state.active_node_id or "",
                    log_snippet=pbq_data.get("log_snippet", ""),
                    student_analysis=student_analysis.strip()
                )
                st.session_state.pbq_results = results
                log_session_to_supabase(
                    unit=st.session_state.active_domain,
                    subtopic=st.session_state.active_subtopic,
                    app_mode="pbq_tester",
                    score_pct=results.get("score", 0),
                    keywords_used=results.get("keywords_used", []),
                    keywords_missed=[],
                    misconception_flag=0 if results.get("score", 0) >= 70 else 1
                )
                st.rerun()

    if st.session_state.pbq_results:
        res = st.session_state.pbq_results
        st.success(f"**PBQ Score: {res.get('score', 0)} / 100**")
        st.info(f"**Examiner Feedback:** {res.get('examiner_feedback', '')}")
        if st.button("Select Another Objective", use_container_width=True):
            reset_session()