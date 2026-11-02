import importlib
import json
import os
import re
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# --- Sidebar Logo Setup ---
LOGO_PATH = "zp_logo.png"

# --- Load Dynamic Course Spec FIRST ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {
        "course_title": "OCR A-Level Computer Science",
        "subject": "Computer Science",
        "level": "A-Level",
        "target_turns": 7,
        "topics": {"General": ["General Practice"]}
    }

COURSE_TITLE = COURSE_SPEC.get("course_title", "OCR A-Level Computer Science")
SUBJECT = COURSE_SPEC.get("subject", "Computer Science")
LEVEL = COURSE_SPEC.get("level", "A-Level")
TARGET_TURNS = COURSE_SPEC.get("target_turns", 7)

# --- Dynamic Page Configuration ---
st.set_page_config(
    page_title=f"{COURSE_TITLE} - ZeroPi Socratic Coach",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🎓",
    layout="wide",
)

if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.sidebar.columns([1, 6, 1])
    with col2:
        st.image(LOGO_PATH, use_container_width=True)

# Ensure API Key is set BEFORE loading CAG FSM (Set both standard env vars)
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
    workflow,
    verify_topic_alignment,
    stream_socratic_response,
    stream_freestyle_socratic_response,
    generate_freestyle_synthesis_prompt,
    generate_socratic_synthesis_prompt,
    evaluate_socratic_synthesis,
    generate_quiz_questions,
    evaluate_quiz_answers,
    generate_extended_question,
    grade_extended_response,
    generate_layman_transformation_prompt,
    grade_disciplinary_rewrite
)

# --- COLOR THEME CONFIGURATION: COMPUTER SCIENCE (Ocean Blue) ---
THEME_PRIMARY = "#0284c7"        # Ocean Blue for primary buttons, borders, and progress bars
THEME_GRADIENT_START = "#0369a1"  # Dark Ocean Blue Header Start
THEME_GRADIENT_END = "#0284c7"    # Ocean Blue Header End
THEME_BANNER_BG = "#f0f9ff"       # Light Sky Purpose Banner
THEME_BANNER_BORDER = "#bae6fd"   # Soft Blue Border
THEME_BANNER_TEXT = "#0369a1"     # Dark Blue Banner Text

# --- Database & Zero-PII Logging Setup ---
Base = declarative_base()

class ActivityLog(Base):
    __tablename__ = "activity_logs"  # Default table for Computer Science

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
    """Logs session metrics statelessly to Supabase activity_logs table."""
    try:
        db_url = st.secrets["postgres"]["url"]
        engine = create_engine(db_url, pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        session = Session()

        kw_used_str = ", ".join([str(k) for k in keywords_used if k]) if keywords_used else None
        kw_missed_str = ", ".join([str(k) for k in keywords_missed if k]) if keywords_missed else None

        log_entry = ActivityLog(
            subject=COURSE_TITLE,
            level=LEVEL,
            unit=unit or "General",
            subtopic=subtopic or "General Practice",
            app_mode=app_mode,
            score_pct=round(float(score_pct), 1),
            keywords_used=kw_used_str,
            keywords_missed=kw_missed_str,
            misconception_flag=int(misconception_flag)
        )

        session.add(log_entry)
        session.commit()
        session.close()
    except Exception as e:
        st.warning(f"⚠️ Telemetry log failed: {e}")

# --- Formatting Helpers ---
def extract_clean_text(response) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return extract_clean_text(response.content)
    if isinstance(response, list) and len(response) > 0:
        first_item = response[0]
        if isinstance(first_item, dict):
            return first_item.get("text", str(first_item))
        elif hasattr(first_item, "text"):
            return first_item.text
        return extract_clean_text(first_item)
    if isinstance(response, dict):
        if "text" in response:
            return response["text"]
        elif "content" in response:
            return extract_clean_text(response["content"])
    return str(response)

def clean_latex(text: str) -> str:
    text = re.sub(r'\$([^\$]+)\$', r'<b>\1</b>', text)
    return text.replace('$', '')

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
    .stApp {{ background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }}
    div[data-testid="stSidebar"] {{ background-color: #ffffff; border-right: 1px solid #e2e8f0; }}
    div[data-testid="stProgress"] > div > div > div {{ background-color: {THEME_PRIMARY} !important; }}
    h1, h2, h3 {{ color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 700; }}
    .chat-header {{ 
        background: linear-gradient(135deg, {THEME_GRADIENT_START} 0%, {THEME_GRADIENT_END} 100%); 
        color: white; padding: 22px; font-weight: 700; text-align: center; 
        font-size: 1.3em; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.25);
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
    .summary-box {{ 
        background: #fefce8; border-left: 5px solid #eab308; padding: 10px 14px; 
        border-radius: 12px; color: #713f12; font-size: 0.93em; margin: 8px 0; max-width: 85%; 
    }}
    .stButton > button {{ border-radius: 12px !important; font-weight: 600 !important; margin-top: 4px !important; margin-bottom: 4px !important; }}
    button[kind="primary"] {{
        background-color: {THEME_PRIMARY} !important;
        border-color: {THEME_PRIMARY} !important;
    }}
    div[data-baseweb="textarea"], 
    div[data-baseweb="textarea"] > div,
    textarea {{
        background-color: #ffffff !important;
    }}
    div[data-baseweb="textarea"] {{
        border: 2px solid {THEME_PRIMARY} !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }}
    div[data-baseweb="textarea"]:focus-within {{
        border-color: {THEME_GRADIENT_START} !important;
        box-shadow: 0 0 0 3px rgba(3, 105, 161, 0.3) !important;
    }}
    textarea {{
        color: #0f172a !important;
        font-size: 1rem !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialisation ---
if "active_unit" not in st.session_state:
    st.session_state.active_unit = None
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "is_freestyle" not in st.session_state:
    st.session_state.is_freestyle = False
if "freestyle_prompt" not in st.session_state:
    st.session_state.freestyle_prompt = ""
if "freestyle_focus" not in st.session_state:
    st.session_state.freestyle_focus = "Technical Explanations & Systems Mechanics"
if "app_mode" not in st.session_state:
    st.session_state.app_mode = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_user_answers" not in st.session_state:
    st.session_state.quiz_user_answers = {}
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = None
if "extended_question" not in st.session_state:
    st.session_state.extended_question = None
if "extended_results" not in st.session_state:
    st.session_state.extended_results = None
if "submitted_extended_answer" not in st.session_state:
    st.session_state.submitted_extended_answer = ""
if "rewrite_data" not in st.session_state:
    st.session_state.rewrite_data = None
if "rewrite_results" not in st.session_state:
    st.session_state.rewrite_results = None
if "student_rewrite_submission" not in st.session_state:
    st.session_state.student_rewrite_submission = ""
if "logged_sessions" not in st.session_state:
    st.session_state.logged_sessions = set()
if "socratic_dialogue_complete" not in st.session_state:
    st.session_state.socratic_dialogue_complete = False
if "socratic_synthesis_q" not in st.session_state:
    st.session_state.socratic_synthesis_q = None
if "socratic_synthesis_results" not in st.session_state:
    st.session_state.socratic_synthesis_results = None

if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [], 
        "sub_topic": st.session_state.active_topic, 
        "turn_count": 0, 
        "is_final_turn": False,
        "is_freestyle": False
    }

def reset_session():
    st.session_state.active_unit = None
    st.session_state.active_topic = None
    st.session_state.is_freestyle = False
    st.session_state.freestyle_prompt = ""
    st.session_state.freestyle_focus = "Technical Explanations & Systems Mechanics"
    st.session_state.app_mode = None
    st.session_state.messages = []
    st.session_state.quiz_questions = None
    st.session_state.quiz_user_answers = {}
    st.session_state.quiz_feedback = None
    st.session_state.extended_question = None
    st.session_state.extended_results = None
    st.session_state.submitted_extended_answer = ""
    st.session_state.rewrite_data = None
    st.session_state.rewrite_results = None
    st.session_state.student_rewrite_submission = ""
    st.session_state.logged_sessions = set()
    st.session_state.socratic_dialogue_complete = False
    st.session_state.socratic_synthesis_q = None
    st.session_state.socratic_synthesis_results = None
    st.session_state.socratic_retry_mode = False
    st.session_state.graph_state = {
        "messages": [], "sub_topic": None, "turn_count": 0, "is_final_turn": False, "is_freestyle": False
    }
    st.rerun()

# --- Dynamic Screen Router ---

if st.session_state.active_topic is None and st.session_state.app_mode != "freestyle_input":
    st.markdown(f'<div class="chat-header">💻 {COURSE_TITLE} Socratic Coach</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="purpose-banner">
            💡 <b>App Purpose:</b> The aim of this app is to help you develop your use of disciplinary literacy. 
            To gain a top grade in Computer Science, you need to think, trace, and write like a Computer Scientist!
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("⚙️ Select Revision Target")
    
    raw_structure = COURSE_SPEC.get("subjects") or COURSE_SPEC.get("topics", {})
    first_key = next(iter(raw_structure), None)
    is_three_tier = isinstance(raw_structure.get(first_key), dict) if first_key else False

    if is_three_tier:
        st.write("Choose a paper, section, and subtopic to begin your practice session:")
        sel_subject = st.selectbox("🖥️ Step 1: Choose Component / Paper:", options=list(raw_structure.keys()))
        units_dict = raw_structure.get(sel_subject, {})
        sel_unit = st.selectbox("📘 Step 2: Choose Section / Unit:", options=list(units_dict.keys()))
        subtopics = units_dict.get(sel_unit, [])
        sel_subtopic = st.selectbox("🔍 Step 3: Choose Specific Subtopic:", options=subtopics)
        
        target_unit_name = sel_unit
        target_subtopic_name = sel_subtopic
        full_query = f"{sel_subject} - {sel_unit}: {sel_subtopic}"
    else:
        st.write("Choose a unit and subtopic to begin your practice session:")
        sel_unit = st.selectbox("📘 Step 1: Choose Unit / Component:", options=list(raw_structure.keys()))
        subtopics = raw_structure.get(sel_unit, [])
        sel_subtopic = st.selectbox("🔍 Step 2: Choose Specific Subtopic:", options=subtopics)
        
        target_unit_name = sel_unit
        target_subtopic_name = sel_subtopic
        full_query = sel_subtopic

    st.write("")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🤖 Start Socratic Target Session", type="primary", use_container_width=True):
            st.session_state.app_mode = "socratic"
            st.session_state.is_freestyle = False
            st.session_state.active_unit = target_unit_name
            st.session_state.active_topic = target_subtopic_name
            st.session_state.graph_state["sub_topic"] = full_query
            st.session_state.graph_state["is_freestyle"] = False
            st.rerun()

    with col_btn2:
        if st.button("🎙️ Start Freestyle CS Practice", type="primary", use_container_width=True):
            st.session_state.app_mode = "freestyle_input"
            st.rerun()

    st.write("---")

    if st.button("📑 Take a Retrieval Quiz", use_container_width=True):
        st.session_state.app_mode = "quiz"
        st.session_state.active_unit = target_unit_name
        st.session_state.active_topic = target_subtopic_name
        with st.spinner("Generating specification retrieval questions..."):
            st.session_state.quiz_questions = generate_quiz_questions(
                full_query, COURSE_TITLE, LEVEL
            )
        st.rerun()

    if st.button("⚡ Answer an Extended Question", use_container_width=True):
        st.session_state.app_mode = "extended"
        st.session_state.active_unit = target_unit_name
        st.session_state.active_topic = target_subtopic_name
        with st.spinner("Generating high-tier extended response scenario..."):
            st.session_state.extended_question = generate_extended_question(full_query, COURSE_TITLE, LEVEL)
        st.rerun()

    if st.button("🛠️ Rewrite an Answer", use_container_width=True):
        st.session_state.app_mode = "rewrite"
        st.session_state.active_unit = target_unit_name
        st.session_state.active_topic = target_subtopic_name
        with st.spinner("Generating informal response scenario..."):
            st.session_state.rewrite_data = generate_layman_transformation_prompt(full_query, COURSE_TITLE, LEVEL)
        st.rerun()

# --- Freestyle Input View ---
elif st.session_state.app_mode == "freestyle_input":
    st.markdown(f'<div class="chat-header">🎙️ Freestyle Computer Science Practice</div>', unsafe_allow_html=True)
    st.subheader("Explore Any Custom CS Algorithm, Concept, or Code Problem")
    st.write("Enter any question, system architecture, or programming concept you'd like to explore Socratically:")

    custom_input = st.text_input(
        "Enter your CS topic, code snippet, or algorithm prompt:",
        placeholder="e.g. How does Dijkstra's algorithm handle priority queues differently than A* search?",
        key="freestyle_custom_input"
    )

    freestyle_focus = st.radio(
        "🎯 Select Practice Focus:",
        ["Technical Explanations & Systems Mechanics", "Key Terminology & Formal Definitions"],
        horizontal=True,
        key="freestyle_focus_selector"
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if st.button("Launch Freestyle Dialogue", type="primary", use_container_width=True):
            if custom_input.strip():
                with st.spinner("Checking specification alignment..."):
                    alignment = verify_topic_alignment(custom_input.strip())

                if alignment.get("is_aligned", True):
                    st.session_state.app_mode = "socratic"
                    st.session_state.is_freestyle = True
                    st.session_state.active_unit = "Freestyle Practice"
                    st.session_state.active_topic = custom_input.strip()
                    st.session_state.freestyle_prompt = custom_input.strip()
                    st.session_state.freestyle_focus = freestyle_focus
                    st.session_state.graph_state["sub_topic"] = custom_input.strip()
                    st.session_state.graph_state["is_freestyle"] = True
                    st.session_state.graph_state["freestyle_focus"] = freestyle_focus
                    st.rerun()
                else:
                    st.error(f"⚠️ **Topic Alignment Guardrail:** {alignment.get('reasoning', 'Topic out of spec scope.')}")
                    if alignment.get("suggested_pivot"):
                        st.info(f"💡 **Suggested Specification Pivot:** {alignment.get('suggested_pivot')}")
            else:
                st.warning("Please enter a question or topic first.")

    with col_f2:
        if st.button("Cancel / Back to Selection Screen", use_container_width=True):
            reset_session()

# --- Socratic Mode View ---
elif st.session_state.app_mode == "socratic":
    st.markdown(f'<div class="chat-header">🤖 {COURSE_TITLE} Coach</div>', unsafe_allow_html=True)
    
    student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")

    with st.sidebar:
        st.subheader("📌 Active Target")
        if st.session_state.is_freestyle:
            st.info(f"**Mode:** Freestyle Practice\n\n**Focus:** {st.session_state.freestyle_focus}\n\n**Topic:** {st.session_state.active_topic}")
            st.metric(label="Turn Counter", value=f"{student_turns} (Uncapped)")
            if student_turns >= 1 and not st.session_state.socratic_dialogue_complete:
                if st.button("🏁 Complete & Take 6-Mark Synthesis", type="primary", use_container_width=True):
                    st.session_state.socratic_dialogue_complete = True
                    st.rerun()
        else:
            st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
            st.metric(label="Turn Counter", value=f"{student_turns} / {TARGET_TURNS}")
            st.progress(min(student_turns / TARGET_TURNS, 1.0))

        st.write("---")
        if st.button("🔄 New Session / Change Topic", use_container_width=True):
            reset_session()

    if len(st.session_state.messages) == 0:
        if st.session_state.is_freestyle:
            if "Terminology" in st.session_state.get("freestyle_focus", ""):
                initial_greeting = (
                    f"Welcome! Let's build your computer science terminology for: **'{st.session_state.active_topic}'**. "
                    f"To start, what key technical terms or definitions relate to this concept?"
                )
            else:
                initial_greeting = (
                    f"Welcome! Let me guide you through: **'{st.session_state.active_topic}'**. "
                    f"To begin, what is the initial operation, state, or execution step involved?"
                )
        else:
            initial_greeting = (
                f"Welcome! We're exploring **{st.session_state.active_topic}** today. "
                f"To get started, what key data structure, algorithm, hardware principle, or theory concept would you like to review?"
            )
        st.session_state.messages.append({"role": "tutor", "content": initial_greeting, "style": "tutor-msg"})
        st.session_state.graph_state["messages"].append(AIMessage(content=initial_greeting))

    for msg in st.session_state.messages:
        html_content = md_to_html(msg["content"])
        if msg["role"] == "tutor":
            div_class = msg.get("style", "tutor-msg")
            header = "🎓 <b>Tutor</b>"
            st.markdown(f'<div class="{div_class}">{header}<br><br>{html_content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="student-msg">🎒 <b>Student</b><br><br>{html_content}</div>', unsafe_allow_html=True)

    if not st.session_state.is_freestyle and student_turns >= TARGET_TURNS and not st.session_state.socratic_dialogue_complete:
        st.info(f"🎉 **Dialogue Phase Complete!** Proceed below to the 6-Mark Synthesis Challenge.")

    is_disabled = (not st.session_state.is_freestyle and student_turns >= TARGET_TURNS) or st.session_state.socratic_dialogue_complete
    placeholder = "Dialogue phase complete. Complete the 6-mark synthesis below." if is_disabled else "Type your response here..."
    
    if user_input := st.chat_input(placeholder, disabled=is_disabled):
        st.session_state.messages.append({"role": "student", "content": user_input})
        st.session_state.graph_state["messages"].append(HumanMessage(content=user_input))
        
        current_student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")
        st.session_state.graph_state["turn_count"] = current_student_turns

        if st.session_state.is_freestyle:
            with st.spinner("Analyzing response and guiding dialogue..."):
                full_stream = ""
                generator = stream_freestyle_socratic_response(
                    messages=st.session_state.graph_state["messages"],
                    custom_prompt=st.session_state.freestyle_prompt,
                    focus_mode=st.session_state.freestyle_focus
                )
                for chunk in generator:
                    text_chunk = extract_clean_text(chunk)
                    full_stream += text_chunk
                
                ai_reply = clean_latex(full_stream)
                st.session_state.messages.append({"role": "tutor", "content": ai_reply, "style": "tutor-msg"})
                st.session_state.graph_state["messages"].append(AIMessage(content=ai_reply))
                st.rerun()
        else:
            st.session_state.graph_state["is_final_turn"] = (current_student_turns >= TARGET_TURNS)
            with st.spinner("Analyzing response and generating Socratic feedback..."):
                full_stream = ""
                generator = stream_socratic_response(
                    messages=st.session_state.graph_state["messages"],
                    sub_topic=st.session_state.active_topic
                )
                for chunk in generator:
                    text_chunk = extract_clean_text(chunk)
                    full_stream += text_chunk

                ai_reply = clean_latex(full_stream)
                st.session_state.messages.append({"role": "tutor", "content": ai_reply, "style": "tutor-msg"})
                st.session_state.graph_state["messages"].append(AIMessage(content=ai_reply))

                if current_student_turns >= TARGET_TURNS:
                    st.session_state.socratic_dialogue_complete = True

                st.rerun()

    # --- Step 2: Render 6-Mark Disciplinary Literacy Synthesis ---
    if st.session_state.socratic_dialogue_complete or (not st.session_state.is_freestyle and student_turns >= TARGET_TURNS):
        st.write("---")
        st.subheader("🎯 Final Mastery Challenge (6 Marks)")
        st.markdown(
            "Apply your disciplinary literacy by synthesizing today's discussion into a coherent technical summary:"
        )

        if st.session_state.socratic_synthesis_q is None:
            with st.spinner("Generating 6-mark synthesis prompt based on your discussion..."):
                if st.session_state.is_freestyle:
                    st.session_state.socratic_synthesis_q = generate_freestyle_synthesis_prompt(
                        custom_prompt=st.session_state.freestyle_prompt,
                        chat_history=st.session_state.messages,
                        focus_mode=st.session_state.freestyle_focus,
                        course_title=COURSE_TITLE,
                        level=LEVEL
                    )
                else:
                    st.session_state.socratic_synthesis_q = generate_socratic_synthesis_prompt(
                        sub_topic=st.session_state.active_topic,
                        chat_history=st.session_state.messages,
                        course_title=COURSE_TITLE,
                        level=LEVEL
                    )

        s_data = st.session_state.socratic_synthesis_q

        # Unsubmitted View
        if st.session_state.socratic_synthesis_results is None:
            st.info(f"**Task Prompt (6 Marks):** {s_data['question']}")
            
            user_synth_ans = st.text_area(
                "Write your technical synthesis here (connect core mechanisms using key terminology):", 
                height=180, 
                key="socratic_synth_input"
            )
            
            if st.button("Submit Synthesis for Evaluation", type="primary", use_container_width=True):
                if user_synth_ans.strip():
                    st.session_state["last_submitted_synthesis"] = user_synth_ans.strip()
                    with st.spinner("Evaluating disciplinary terminology and conceptual coherence..."):
                        eval_res = evaluate_socratic_synthesis(
                            question=s_data['question'],
                            mark_scheme=s_data['mark_scheme'],
                            student_answer=user_synth_ans.strip(),
                            chat_history=st.session_state.messages,
                            course_title=COURSE_TITLE,
                            level=LEVEL
                        )
                        st.session_state.socratic_synthesis_results = eval_res
                        
                        score_out_of_6 = eval_res.get("score", 0)
                        score_pct = (score_out_of_6 / 6.0) * 100.0
                        
                        if "socratic_completed" not in st.session_state.logged_sessions:
                            log_session_to_supabase(
                                unit=st.session_state.active_unit,
                                subtopic=st.session_state.active_topic,
                                app_mode="freestyle_socratic" if st.session_state.is_freestyle else "socratic",
                                score_pct=score_pct,
                                keywords_used=eval_res.get("keywords_used", []),
                                keywords_missed=eval_res.get("keywords_missed", []),
                                misconception_flag=1 if score_pct < 45.0 else 0
                            )
                            st.session_state.logged_sessions.add("socratic_completed")
                        
                        st.rerun()
                else:
                    st.warning("Please enter your answer before submitting.")

        # Post-Submission Marked View
        else:
            res = st.session_state.socratic_synthesis_results
            current_score = res.get("score", 0)
            
            with st.expander("📌 View Task Prompt & Your Response", expanded=True):
                st.markdown(f"**Task Prompt (6 Marks):** {s_data['question']}")
                st.markdown(f"**Your Answer:**\n\n> {st.session_state.get('last_submitted_synthesis', '')}")

            st.success(f"🏆 **Synthesis Score: {current_score} / 6 Marks**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### ✅ Marks Awarded")
                for pt in res.get("awarded_points", []):
                    st.markdown(f"• {pt}")
            with col_b:
                st.markdown("### ❌ Missed Mark Points")
                for pt in res.get("missed_points", []):
                    st.markdown(f"• {pt}")
                    
            st.info(f"💡 **Examiner Note:** {res.get('feedback', '')}")
            st.write("")

            if current_score < 6:
                if not st.session_state.get("socratic_retry_mode", False):
                    if st.button("🛠️ Upgrade Response to 6/6 (Retry Challenge)", type="primary", use_container_width=True):
                        st.session_state.socratic_retry_mode = True
                        st.rerun()
                else:
                    st.markdown("---")
                    st.subheader("✏️ Upgrade Your Synthesis")
                    st.caption("Refine your previous answer by incorporating the missing technical terms and mechanisms highlighted above:")
                    
                    retry_text = st.text_area(
                        "Revised Response:", 
                        value=st.session_state.get("last_submitted_synthesis", ""), 
                        height=180, 
                        key="socratic_synth_retry_input"
                    )
                    
                    if st.button("Submit Upgraded Response for Re-Evaluation", type="primary", use_container_width=True):
                        if retry_text.strip():
                            st.session_state["last_submitted_synthesis"] = retry_text.strip()
                            with st.spinner("Re-evaluating upgraded response..."):
                                eval_res = evaluate_socratic_synthesis(
                                    question=s_data['question'],
                                    mark_scheme=s_data['mark_scheme'],
                                    student_answer=retry_text.strip(),
                                    chat_history=st.session_state.messages,
                                    course_title=COURSE_TITLE,
                                    level=LEVEL
                                )
                                st.session_state.socratic_synthesis_results = eval_res
                                st.session_state.socratic_retry_mode = False
                                
                                new_score = eval_res.get("score", 0)
                                new_pct = (new_score / 6.0) * 100.0
                                log_session_to_supabase(
                                    unit=st.session_state.active_unit,
                                    subtopic=st.session_state.active_topic,
                                    app_mode="socratic_retry",
                                    score_pct=new_pct,
                                    keywords_used=eval_res.get("keywords_used", []),
                                    keywords_missed=eval_res.get("keywords_missed", []),
                                    misconception_flag=1 if new_pct < 45.0 else 0
                                )
                                st.rerun()

            st.write("")
            if st.button("🔄 Start New Session / Select Topic", use_container_width=True):
                reset_session()

# --- Quiz Mode View ---
elif st.session_state.app_mode == "quiz":
    st.markdown(f'<div class="chat-header">📑 {COURSE_TITLE} Retrieval Quiz</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        st.write("---")
        if st.button("🔄 Change Topic / Mode", use_container_width=True):
            reset_session()

    questions = st.session_state.get("quiz_questions")
    
    if questions:
        if st.session_state.quiz_feedback is None:
            with st.form("retrieval_quiz_form"):
                st.subheader(f"Practice Quiz: {st.session_state.active_topic}")
                user_answers = {}
                for idx, q in enumerate(questions, 1):
                    q_text = q.get("question", q) if isinstance(q, dict) else q
                    st.markdown(f"**Q{idx}: {q_text}**")
                    user_answers[idx] = st.text_input(f"Your Answer for Q{idx}:", key=f"quiz_ans_{idx}")
                    st.write("")
                
                submitted = st.form_submit_button("Submit Quiz for Feedback", type="primary", use_container_width=True)
                
                if submitted:
                    st.session_state.quiz_user_answers = user_answers
                    with st.spinner("Evaluating your responses against specification mark schemes..."):
                        feedback = evaluate_quiz_answers(
                            questions=questions,
                            user_answers=user_answers,
                            topic=st.session_state.active_topic,
                            course_title=COURSE_TITLE,
                            level=LEVEL
                        )
                        st.session_state.quiz_feedback = feedback
                        
                        raw_score = feedback.get("total_score", 0) if isinstance(feedback, dict) else 0
                        try:
                            total_score = int(raw_score)
                        except (ValueError, TypeError):
                            total_score = 0

                        score_pct = (total_score / len(questions)) * 100.0 if questions else 0.0
                        
                        all_used, all_missed = [], []
                        for item in feedback.get("breakdown", []):
                            all_used.extend(item.get("keywords_used", []))
                            all_missed.extend(item.get("keywords_missed", []))
                            
                        log_session_to_supabase(
                            unit=st.session_state.active_unit,
                            subtopic=st.session_state.active_topic,
                            app_mode="quiz",
                            score_pct=score_pct,
                            keywords_used=all_used,
                            keywords_missed=all_missed,
                            misconception_flag=1 if score_pct < 45.0 else 0
                        )
                        st.rerun()
        else:
            feedback_data = st.session_state.quiz_feedback
            raw_score = feedback_data.get("total_score", 0) if isinstance(feedback_data, dict) else 0
            try:
                total_score = int(raw_score)
            except (ValueError, TypeError):
                total_score = 0

            breakdown = feedback_data.get("breakdown", []) if isinstance(feedback_data, dict) else []
            total_questions = len(breakdown) if breakdown else len(questions)

            st.success(f"🎉 **Quiz Complete! Total Score: {total_score} / {total_questions}**\n\nReview your keyword accuracy breakdown below:")
            st.write("")

            stored_answers = st.session_state.get("quiz_user_answers", {})

            for idx, item in enumerate(breakdown, 1):
                q_num = item.get("question_num", idx)
                
                # Retrieve question text safely
                if idx <= len(questions):
                    orig_q = questions[idx - 1]
                    q_text = orig_q.get("question", orig_q) if isinstance(orig_q, dict) else orig_q
                else:
                    q_text = item.get("question", f"Question {q_num}")

                score = item.get("score", 0)
                user_ans = item.get("student_answer", stored_answers.get(q_num, "No answer provided"))
                model_ans = item.get("model_answer", item.get("feedback", ""))
                used = ", ".join(item.get("keywords_used", [])) or "None"
                missed = ", ".join(item.get("keywords_missed", [])) or "None"
                examiner_note = item.get("examiner_note", item.get("explanation", item.get("feedback", "")))

                label = f"Q{q_num}: {q_text} — Score: {score}/1"
                
                with st.expander(label, expanded=False):
                    st.markdown(f"**Your Answer:**\n\n> {user_ans}")
                    st.markdown(f"**Model Answer:** {model_ans}")
                    st.markdown(f"**Key Terms Used:** {used}")
                    st.markdown(f"**Missed Keywords:** {missed}")
                    st.info(f"💡 **Examiner Note:** {examiner_note}")

            st.write("")
            if st.button("🔄 Retake Quiz / Try Another Topic", type="primary", use_container_width=True):
                reset_session()
    else:
        st.error("No questions were generated. Please return and select a topic again.")
        if st.button("Back to Selection Screen"):
            reset_session()

# --- Extended Question Mode View ---
elif st.session_state.app_mode == "extended":
    st.markdown(f'<div class="chat-header">⚡ {COURSE_TITLE} Extended Question</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        st.write("---")
        if st.button("🔄 Change Topic / Mode", use_container_width=True):
            reset_session()

    q_text = st.session_state.get("extended_question")
    
    if q_text:
        if st.session_state.extended_results is None:
            st.subheader("Extended Response Challenge")
            st.markdown(f"**Question:** {q_text}")
            
            user_response = st.text_area("Write your detailed evaluation below (include trade-off analysis and technical terms):", height=200)
            
            if st.button("Submit Extended Answer", type="primary", use_container_width=True):
                if user_response.strip():
                    st.session_state.submitted_extended_answer = user_response.strip()
                    with st.spinner("Evaluating disciplinary literacy and complexity analysis..."):
                        results = grade_extended_response(
                            sub_topic=st.session_state.active_topic,
                            question=q_text,
                            student_answer=user_response.strip(),
                            course_title=COURSE_TITLE,
                            level=LEVEL
                        )
                        st.session_state.extended_results = results
                        
                        score = results.get("score", 0)
                        max_score = results.get("max_score", 9)
                        score_pct = (score / max_score) * 100.0 if max_score > 0 else 0.0
                        
                        log_session_to_supabase(
                            unit=st.session_state.active_unit,
                            subtopic=st.session_state.active_topic,
                            app_mode="extended",
                            score_pct=score_pct,
                            keywords_used=results.get("keywords_used", []),
                            keywords_missed=results.get("keywords_missed", []),
                            misconception_flag=1 if score_pct < 45.0 else 0
                        )
                        st.rerun()
                else:
                    st.warning("Please type an answer before submitting.")
        else:
            res = st.session_state.extended_results
            
            with st.expander("📝 Your Submitted Extended Answer", expanded=True):
                st.markdown(f"**Question:** {q_text}")
                st.markdown(f"**Your Answer:**\n\n> {st.session_state.get('submitted_extended_answer', '')}")

            st.success(f"🎉 **Evaluation Complete! Score: {res.get('score', 0)} / {res.get('max_score', 9)} ({res.get('disciplinary_level', 'Developing')})**")
            
            st.markdown(f"**Strengths:** {res.get('strengths', '')}")
            st.markdown(f"**Key Terms Used:** {', '.join(res.get('keywords_used', [])) or 'None'}")
            st.markdown(f"**Missed Key Terms:** {', '.join(res.get('keywords_missed', [])) or 'None'}")
            st.info(f"💡 **Advice for Improvement:** {res.get('struggle_advice', '')}")
            
            with st.expander("📖 View Exemplar Top-Band Model Answer"):
                st.markdown(res.get("model_answer", ""))
                
            st.write("")
            if st.button("🔄 Try Another Question / Topic", type="primary", use_container_width=True):
                reset_session()

# --- Rewrite Mode View ---
elif st.session_state.app_mode == "rewrite":
    st.markdown(f'<div class="chat-header">🛠️ {COURSE_TITLE} Disciplinary Rewrite</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        st.write("---")
        if st.button("🔄 Change Topic / Mode", use_container_width=True):
            reset_session()

    r_data = st.session_state.get("rewrite_data")
    
    if r_data:
        if st.session_state.rewrite_results is None:
            st.subheader("Upgrade Informal Answer Challenge")
            st.markdown(f"**Question:** {r_data.get('question')}")
            st.warning(f"**Informal Student Draft:**\n\n\"{r_data.get('layman_answer')}\"")
            
            student_rewrite = st.text_area("Rewrite this response using precise Computer Science terminology and complexity notation:", height=180)
            
            if st.button("Submit Upgraded Rewrite", type="primary", use_container_width=True):
                if student_rewrite.strip():
                    st.session_state.student_rewrite_submission = student_rewrite.strip()
                    with st.spinner("Grading terminology upgrade..."):
                        results = grade_disciplinary_rewrite(
                            sub_topic=st.session_state.active_topic,
                            question=r_data.get('question'),
                            layman_answer=r_data.get('layman_answer'),
                            student_rewrite=student_rewrite.strip(),
                            course_title=COURSE_TITLE,
                            level=LEVEL
                        )
                        st.session_state.rewrite_results = results
                        
                        score = results.get("score", 0)
                        max_score = results.get("max_score", 4)
                        score_pct = (score / max_score) * 100.0 if max_score > 0 else 0.0
                        
                        log_session_to_supabase(
                            unit=st.session_state.active_unit,
                            subtopic=st.session_state.active_topic,
                            app_mode="rewrite",
                            score_pct=score_pct,
                            keywords_used=results.get("key_terms_used", []),
                            keywords_missed=results.get("missed_terms", []),
                            misconception_flag=1 if score_pct < 45.0 else 0
                        )
                        st.rerun()
                else:
                    st.warning("Please type a rewrite before submitting.")
        else:
            res = st.session_state.rewrite_results
            st.success(f"🎉 **Rewrite Graded! Score: {res.get('score', 0)} / {res.get('max_score', 4)}**")
            
            with st.expander("📌 View Question & Your Submission", expanded=True):
                st.markdown(f"**Question:** {r_data.get('question')}")
                st.markdown(f"**Original Informal Draft:** *\"{r_data.get('layman_answer')}\"*")
                st.markdown(f"**Your Upgraded Rewrite:**\n\n> {st.session_state.student_rewrite_submission}")
            
            st.write("")
            st.markdown(f"**Technical Terms Added:** {', '.join(res.get('key_terms_used', [])) or 'None'}")
            st.markdown(f"**Missed Terms:** {', '.join(res.get('missed_terms', [])) or 'None'}")
            st.info(f"💡 **Examiner Note:** {res.get('feedback', '')}")
            
            st.write("")
            if st.button("🔄 Try Another Rewrite / Topic", type="primary", use_container_width=True):
                reset_session()