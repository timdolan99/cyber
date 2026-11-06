import os
import json
import re
import sys
from pathlib import Path
from typing import List, TypedDict, Dict, Any, Generator

from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
from pypdf import PdfReader

# Use built-in tomllib for Python 3.11+ or fallback to tomli/custom parsing
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# --- Flexible API Key Loader ---
def load_api_key() -> str:
    """Load API key from .streamlit/secrets.toml, secrets directory, or environment variables."""
    base_dir = Path(__file__).resolve().parent
    
    streamlit_secret_paths = [
        base_dir / ".streamlit" / "secrets.toml",
        base_dir.parent / ".streamlit" / "secrets.toml"
    ]
    
    for secret_path in streamlit_secret_paths:
        if secret_path.is_file():
            if tomllib:
                try:
                    with open(secret_path, "rb") as f:
                        data = tomllib.load(f)
                        key = (
                            data.get("GEMINI_API_KEY") or 
                            data.get("GOOGLE_API_KEY") or 
                            data.get("api_key") or
                            (data.get("gemini") and data["gemini"].get("api_key"))
                        )
                        if key:
                            return key.strip()
                except Exception:
                    pass
            else:
                for line in secret_path.read_text(encoding="utf-8").splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.split("=", 1)
                        if k.strip().upper() in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY"]:
                            return v.strip().strip('"').strip("'")

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Could not locate a valid Gemini API key in .streamlit/secrets.toml or environment variables.")
    return key

# --- Initialize Google GenAI Client ---
client = genai.Client(api_key=load_api_key())

# --- Load Dynamic CompTIA Course Specification & Nodes ---
BASE_DIR = Path(__file__).resolve().parent
SPEC_PATH = BASE_DIR / "comptia_secplus_spec.json"
NODES_PATH = BASE_DIR / "comptia_nodes.json"

if SPEC_PATH.is_file():
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {}

if NODES_PATH.is_file():
    with open(NODES_PATH, "r", encoding="utf-8") as f:
        COMPTIA_NODES = json.load(f)
else:
    COMPTIA_NODES = []

COURSE_TITLE = COURSE_SPEC.get("course_title", "CompTIA Security+")
EXAM_CODE = COURSE_SPEC.get("code", "SY0-701")
MODEL_NAME = "gemini-3.8-flash"

# Module-level memory cache for exam pool generation efficiency
_CACHED_EXAM_POOL = None

# Global cache for parsed syllabus documents to eliminate disk I/O lag
_CACHED_SYLLABUS_TEXT = None


class ChatState(TypedDict, total=False):
    messages: List[BaseMessage]
    node_id: str
    sub_topic: str
    turn_count: int
    is_validated: bool
    is_freestyle: bool
    freestyle_focus: str


def get_node_by_id(node_id: str) -> Dict[str, Any]:
    """Retrieves node metadata from comptia_nodes.json."""
    for node in COMPTIA_NODES:
        if node.get("node_id") == node_id:
            return node
    return {}


# --- Fast Local Terminology Pre-Checker (<10ms Execution) ---
def check_terms_locally(user_input: str, node_id: str) -> Dict[str, Any]:
    """Evaluates candidate response locally in <10ms against node metadata without API calls."""
    node = get_node_by_id(node_id)
    if not node:
        return {"matched_terms": [], "matched_acronyms": [], "coverage_pct": 0.0}

    input_lower = user_input.lower()
    
    matched_terms = [t for t in node.get("key_terms", []) if t.lower() in input_lower]
    matched_acronyms = [a for a in node.get("required_acronyms", []) if re.search(r'\b' + re.escape(a.lower()) + r'\b', input_lower)]

    total_req = len(node.get("key_terms", [])) + len(node.get("required_acronyms", []))
    total_found = len(matched_terms) + len(matched_acronyms)
    coverage = (total_found / total_req * 100.0) if total_req > 0 else 100.0

    return {
        "matched_terms": matched_terms,
        "matched_acronyms": matched_acronyms,
        "coverage_pct": round(coverage, 1)
    }


# --- Mode-Specific Suggested Prompt Generator ---
def get_suggested_prompts(mode: str, node_id: str = "") -> Dict[str, Any]:
    """Returns intuitive, mode-specific exemplar prompts to guide user prompting."""
    node = get_node_by_id(node_id)
    sub_obj = node.get("sub_objective", "selected objective")
    acronyms = ", ".join(node.get("required_acronyms", ["MFA", "AES-256"]))
    
    if mode == "socratic_drill":
        return {
            "mode_title": "Guided Socratic Drill",
            "instruction": "State your initial explanation using precise technical terms, mandatory acronyms, and control mechanisms.",
            "suggested_prompt": (
                f"For {sub_obj}, my understanding is that we enforce confidentiality in transit using "
                f"symmetric encryption algorithms like AES-256, while integrity is verified using "
                f"cryptographic hashes such as SHA-256. We handle access via {acronyms}. "
                f"How does this control framework defend against active man-in-the-middle attacks?"
            ),
            "tips": [
                "Mention exact algorithms (AES-256, SHA-256) rather than generic words like 'encryption' or 'hashing'.",
                "Explicitly link each control mechanism to its CIA Triad pillar (Confidentiality, Integrity, Availability).",
                "Name exact protocol ports or infrastructure components."
            ]
        }
    
    elif mode == "freestyle":
        return {
            "mode_title": "Freestyle Scenario Practice",
            "instruction": "Formulate a real-world enterprise scenario or incident response query for open Socratic exploration.",
            "suggested_prompt": (
                "I am investigating a suspected Pass-the-Hash attack on our Windows Server 2022 domain controller. "
                "The NTLM hash was extracted from LSASS memory. How should I isolate the host using our EDR, "
                "remediate the compromised Kerberos ticket granting tickets (TGT), and update our GPO baseline "
                "to enforce Credential Guard and prevent lateral movement?"
            ),
            "tips": [
                "Describe the threat actor TTP or attack vector clearly (e.g., Pass-the-Hash, Password Spraying, SQLi).",
                "Identify specific operating system components or security tools (LSASS, EDR, GPO, WAF, SIEM).",
                "Ask how to balance containment, forensic preservation, and operational restoration."
            ]
        }

    elif mode == "response_refactor":
        flawed = node.get("refactor_sample_flawed_text", "We hash our passwords using AES so no one can read them.")
        return {
            "mode_title": "Response Refactor",
            "instruction": "Identify the flawed reasoning in the draft response and rewrite it using exact CompTIA terminology.",
            "suggested_prompt": (
                f"Draft to refactor: '{flawed}'\n\n"
                "Refactored Answer: This initial statement incorrectly identifies AES as a hashing algorithm and confuses hashing with encryption. "
                "To ensure confidentiality for data at rest or in transit, we use symmetric ciphers like AES-256. "
                "To protect stored passwords, we apply salted one-way cryptographic hash functions (such as PBKDF2 or bcrypt) "
                "to protect integrity and prevent rainbow table attacks."
            ),
            "tips": [
                "State explicitly why the original draft is technically inaccurate.",
                "Replace informal phrases with official Security+ terminology.",
                "Explain the correct operational implementation."
            ]
        }

    elif mode == "pbq_tester":
        log_sample = node.get("pbq_sample_log_snippet", "192.168.1.15 -> 10.0.0.5 TCP 443 [SYN]")
        return {
            "mode_title": "Text-Based PBQ Log Analysis",
            "instruction": "Analyze the log snippet below, name the attack vector, specify the affected OSI layer/protocol, and detail the technical remediation.",
            "suggested_prompt": (
                f"Log snippet:\n{log_sample}\n\n"
                "Analysis:\n"
                "1. Attack Identification: This pattern represents an amplification or scanning attempt targeting specific open ports.\n"
                "2. Layer & Protocol: Operating at OSI Layer 4 (Transport Layer) using TCP flags.\n"
                "3. Remediation: Implement a rate-limiting rule on the perimeter Next-Generation Firewall (NGFW), "
                "update the Access Control List (ACL) to block rogue source IPs, and enable SYN flood protection on the switch."
            ),
            "tips": [
                "Structure your analysis into 3 clear headings: Attack Vector, OSI Layer/Protocol, and Technical Remediation.",
                "Quote key fields directly from the log payload (IP addresses, ports, HTTP status codes, flags).",
                "Recommend hardware-level or software-level mitigation controls (ACLs, WAF rules, GPO settings)."
            ]
        }

    return {}


# --- Direct System Context Loader (Globally Cached to Eliminate Disk I/O Lag) ---
def get_system_context() -> str:
    """Loads and caches course specs and syllabus docs globally to eliminate disk I/O lag."""
    global _CACHED_SYLLABUS_TEXT
    if _CACHED_SYLLABUS_TEXT is not None:
        return _CACHED_SYLLABUS_TEXT

    spec_content = json.dumps(COURSE_SPEC, indent=2) if COURSE_SPEC else "CompTIA Security+ SY0-701 Spec"
    nodes_content = json.dumps(COMPTIA_NODES, indent=2) if COMPTIA_NODES else "CompTIA Nodes Repository"

    syllabus_docs = []
    syllabus_dir = BASE_DIR / "syllabus"
    
    if syllabus_dir.exists():
        for file_path in syllabus_dir.glob("*"):
            if file_path.suffix.lower() == ".pdf":
                try:
                    reader = PdfReader(file_path)
                    pdf_text = [page.extract_text() for page in reader.pages if page.extract_text()]
                    if pdf_text:
                        syllabus_docs.append(f"--- DOCUMENT: {file_path.name} ---\n" + "\n".join(pdf_text))
                except Exception:
                    pass
            elif file_path.suffix.lower() in [".json", ".txt", ".md"]:
                syllabus_docs.append(f"--- DOCUMENT: {file_path.name} ---\n" + file_path.read_text(encoding="utf-8"))

    combined_syllabus_text = "\n\n".join(syllabus_docs)

    _CACHED_SYLLABUS_TEXT = f"""
    You are an expert CompTIA Security+ ({EXAM_CODE}) Socratic Engine and Lead Cyber Specialist.
    
    === COURSE SPECIFICATION ===
    {spec_content}
    
    === FSM NODE DEFINITIONS & MISCONCEPTIONS ===
    {nodes_content}

    === OFFICIAL SYLLABUS, PRACTICE QUESTION SETS & EXAM GUIDES ===
    {combined_syllabus_text}
    
    PEDAGOGICAL DIRECTIVE:
    Target adult cybersecurity professionals. Demand exact technical precision in terminology. Maintain an uncapped Socratic interaction until the student accurately proves mastery. Keep questions concise and rigorous.
    """
    return _CACHED_SYLLABUS_TEXT


def extract_clean_text(response) -> str:
    """Extracts clean text strings from LLM response payload objects."""
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return extract_clean_text(response.content)
    return str(response)


def convert_messages_to_contents(messages: list) -> list:
    """Efficiently converts chat message lists into Google GenAI content format."""
    contents = []
    for msg in messages:
        if isinstance(msg, dict):
            role = "user" if msg.get("role") in ["user", "student"] else "model"
            content_text = msg.get("content", "")
        elif isinstance(msg, BaseMessage):
            role = "user" if isinstance(msg, HumanMessage) else "model"
            content_text = msg.content
        else:
            continue
            
        if content_text:
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=content_text)]))
            
    return contents


# --- High-Speed Uncapped Socratic Engine Streaming (Direct Context) ---
def stream_socratic_response(messages: list, sub_topic: str, node_id: str = ""):
    node = get_node_by_id(node_id)
    kt, ac = ", ".join(node.get("key_terms", [])), ", ".join(node.get("required_acronyms", []))
    
    directive = (
        f"[ACTIVE SOCRATIC DRILL DIRECTIVE - MULTI-TURN PERSISTENCE]\n"
        f"Target Revision Objective: {sub_topic}\n"
        f"Required Key Terms: {kt}\n"
        f"Required Acronyms: {ac}\n\n"
        f"MANDATORY BEHAVIORAL RULES:\n"
        f"1. This is an ongoing, deep Socratic dialogue. DO NOT conclude the session or provide final grading yet.\n"
        f"2. Keep the Socratic pressure active. Probe deeper into operational mechanics, edge cases, or failure modes.\n"
        f"3. MANDATORY: Wrap every instance of these key terms/acronyms in markdown bold (e.g., **AUP**, **ACL**).\n"
        f"4. Provide 3-4 sentences covering accuracy, positives, gaps, and a closing challenge question. Complete all sentences.\n\n"
    )
    
    system_instruction = get_system_context()
    contents = convert_messages_to_contents(messages)
    
    if contents and contents[-1].role == "user":
        original_text = contents[-1].parts[0].text
        contents[-1].parts[0].text = directive + original_text
    else:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=directive)]))
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.3, 
        max_output_tokens=2048
    )
    
    for chunk in client.models.generate_content_stream(model=MODEL_NAME, contents=contents, config=config):
        if chunk.text: 
            yield chunk.text

# --- Freestyle Practice Mode Helpers ---
def stream_freestyle_socratic_response(
    messages: list, 
    custom_prompt: str, 
    focus_mode: str
) -> Generator[str, None, None]:
    """Streams Socratic guidance for unscripted freestyle practice with full token headroom."""
    directive = (
        f"[EXAMINER DIRECTIVE - FREESTYLE SOCRATIC MODE]\n"
        f"Unscripted Practice Prompt: \"{custom_prompt}\"\n"
        f"Focus: {focus_mode}\n\n"
        f"Provide substantive, analytical feedback (3-4 sentences) evaluating the candidate's custom scenario response, "
        f"challenging inaccuracies, and concluding with a sharp probing question without giving away the answer. "
        f"Complete all sentences fully.\n\n"
    )

    system_instruction = get_system_context()
    contents = convert_messages_to_contents(messages)
    
    if contents and contents[-1].role == "user":
        original_text = contents[-1].parts[0].text
        contents[-1].parts[0].text = directive + original_text
    else:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=directive)]))

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.3,
        max_output_tokens=2048
    )

    response_stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=config
    )

    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


def verify_topic_alignment(custom_prompt: str) -> Dict[str, Any]:
    """Verifies whether a user-entered freestyle prompt aligns with CompTIA Security+ / Cybersecurity."""
    system_instruction = get_system_context()
    system_prompt = f"""
    You are an AI curriculum alignment filter for {COURSE_TITLE} ({EXAM_CODE}).
    Analyze the user input and determine if it falls within the scope of cybersecurity, enterprise architecture, threats, or security operations.

    Return ONLY a valid JSON object:
    {{
      "is_aligned": true/false,
      "reasoning": "Short explanation",
      "suggested_pivot": "Suggested topic in specification if off-topic, else empty"
    }}
    """
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0
    )

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]),
        types.Content(role="user", parts=[types.Part.from_text(text=custom_prompt)])
    ]
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=config
    )
    
    raw_text = extract_clean_text(response)
    clean_json = re.sub(r'```json\s*|\s*```', '', raw_text).strip()
    try:
        return json.loads(clean_json)
    except Exception:
        return {"is_aligned": True, "reasoning": "Bypassed filter verification.", "suggested_pivot": ""}


# --- Response Refactor Feature ---
def generate_refactor_challenge(node_id: str) -> Dict[str, str]:
    """Retrieves or generates a flawed technical response for the student to rewrite."""
    node = get_node_by_id(node_id)
    if node and "refactor_sample_flawed_text" in node:
        return {
            "node_id": node_id,
            "sub_objective": node["sub_objective"],
            "flawed_text": node["refactor_sample_flawed_text"],
            "reference_solution": node.get("refactor_sample_corrected_text", "")
        }

    system_instruction = get_system_context()
    prompt = f"Generate a flawed, vague response written by an amateur candidate for sub-objective '{node_id}'. Return JSON with 'flawed_text' and 'reference_solution'."
    
    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.3, max_output_tokens=2048)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)
    
    clean_text = re.sub(r"```json|```", "", extract_clean_text(response)).strip()
    return json.loads(clean_text)


def grade_refactored_response(node_id: str, flawed_text: str, student_refactor: str) -> Dict[str, Any]:
    """Evaluates the student's refactored response for terminology and conceptual accuracy."""
    system_instruction = get_system_context()
    node = get_node_by_id(node_id)

    prompt = f"""
    Target Objective: {node.get('sub_objective', node_id)}
    Required Acronyms: {', '.join(node.get('required_acronyms', []))}
    Core Key Terms: {', '.join(node.get('key_terms', []))}

    Flawed Initial Draft:
    "{flawed_text}"

    Student Refactored Response:
    "{student_refactor}"

    Evaluate whether the student corrected the flawed concepts and used accurate Security+ terminology.

    Return ONLY a JSON object:
    {{
        "is_correct": true/false,
        "score": 85,
        "matched_acronyms": ["AES", "SHA-256"],
        "missed_terms": ["Integrity"],
        "feedback": "Detailed examiner feedback explaining what was fixed and what needs improvement."
    }}
    """

    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0, max_output_tokens=2048)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)
    
    clean_text = re.sub(r"```json|```", "", extract_clean_text(response)).strip()
    return json.loads(clean_text)


# --- Text-Based PBQ Tester Mode ---
def generate_pbq_scenario(node_id: str) -> Dict[str, Any]:
    """Generates a text-based Performance-Based Question (PBQ) featuring raw logs or configs."""
    node = get_node_by_id(node_id)
    log_snippet = node.get("pbq_sample_log_snippet", "LOG_ALERT: 192.168.1.15 -> 10.0.0.5 TCP 443 [SYN]")

    return {
        "node_id": node_id,
        "sub_objective": node.get("sub_objective", ""),
        "log_snippet": log_snippet,
        "prompt": f"Analyze the log snippet above for sub-objective '{node.get('sub_objective', node_id)}'. Identify the specific attack vector/anomaly, state the affected OSI layer/protocol, and detail the exact technical mitigation step required."
    }


def grade_pbq_response(node_id: str, log_snippet: str, student_analysis: str) -> Dict[str, Any]:
    """Grades student log analysis against CompTIA objectives."""
    system_instruction = get_system_context()
    node = get_node_by_id(node_id)

    prompt = f"""
    Sub-Objective: {node.get('sub_objective', node_id)}
    Log Snippet Provided:
    {log_snippet}

    Student Troubleshooting Response:
    {student_analysis}

    Grade the student's identification of the attack vector, protocol analysis, and mitigation steps against CompTIA Security+ standards.

    Return ONLY JSON:
    {{
        "score": 90,
        "max_score": 100,
        "attack_vector_identified": true/false,
        "correct_mitigation": true/false,
        "keywords_used": ["ACL", "Port 22"],
        "examiner_feedback": "Detailed feedback on the analysis..."
    }}
    """

    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0, max_output_tokens=2048)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)
    
    clean_text = re.sub(r"```json|```", "", extract_clean_text(response)).strip()
    return json.loads(clean_text)


# --- Fully Dynamic 45-Question Exam Pool Generator (Granular Domains & Open-Ended PBQs) ---
def load_questions_from_syllabus_and_nodes() -> list:
    """Generates and memoizes a balanced 45-question exam pool: 2 open-ended PBQs, 10 acronyms/definitions, 33 MCQs."""
    global _CACHED_EXAM_POOL
    if _CACHED_EXAM_POOL is not None:
        return _CACHED_EXAM_POOL

    pool = []
    
    try:
        system_instruction = get_system_context()
        prompt = f"""
        You are an expert CompTIA Security+ ({EXAM_CODE}) senior certification examiner.
        Generate a comprehensive, precisely balanced set of 45 mock exam items.
        
        The 45 items MUST strictly follow this distribution:
        - Exactly 2 Front-Loaded Performance-Based Questions (PBQs) (Type: 'pbq'). Each PBQ must include a realistic raw log snippet or technical incident description (`log_snippet`) and an open-ended troubleshooting prompt.
        - 10 Dedicated Acronym and Definition Identification Questions (Type: 'mcq') focusing on terms like SIEM, HSM, LDAP, PKI, RPO/RTO.
        - 33 Core Domain Multiple-Choice Questions (MCQs) distributed evenly across official CompTIA Security+ domains:
          - Domain 1.0: General Security Concepts
          - Domain 2.0: Threats, Vulnerabilities & Mitigations
          - Domain 3.0: Security Architecture
          - Domain 4.0: Security Operations
          - Domain 5.0: Governance, Risk & Compliance

        Return ONLY a valid JSON list of objects with this exact structure:
        [
          {{
            "type": "mcq" or "pbq",
            "domain": "1.0 General Security Concepts" (Must start with official prefix 1.0, 2.0, 3.0, 4.0, or 5.0),
            "sub_objective": "Specific sub-objective description",
            "question": "The question or troubleshooting prompt text...",
            "log_snippet": "RAW LOG OR CONFIG PAYLOAD (Only required if type is 'pbq', otherwise omit)",
            "options": ["Option A", "Option B", "Option C", "Option D"] (Only required if type is 'mcq'),
            "correct_index": 0 (Only required if type is 'mcq')
          }}
        ]
        """
        config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.3, max_output_tokens=8192)
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)
        clean_json = re.sub(r"```json|```", "", extract_clean_text(response)).strip()
        generated_items = json.loads(clean_json)
        if isinstance(generated_items, list):
            pool.extend(generated_items)
    except Exception as e:
        print(f"Error generating dynamic questions: {e}")

    # Fallback if empty
    if not pool:
        pool.append({
            "type": "pbq",
            "domain": "4.0 Security Operations",
            "sub_objective": "Log Analysis & Incident Response",
            "question": "Analyze the log snippet below, identify the attack vector, and detail technical remediation.",
            "log_snippet": "LOG_ALERT: 192.168.1.50 -> 10.0.0.15 TCP 3389 [SYN-Flood/Brute-Force]",
        })
        pool.append({
            "type": "pbq",
            "domain": "2.0 Threats, Vulnerabilities & Mitigations",
            "sub_objective": "Application Vulnerability Analysis",
            "question": "Analyze the following web server log snippet for suspicious API endpoint parameter tampering.",
            "log_snippet": "GET /api/v1/users?id=OR 1=1 HTTP/1.1 -- Status: 200 OK (Data Dump)",
        })
        for node in COMPTIA_NODES[:43]:
            sub_obj = node.get("sub_objective", "Security Objective")
            pool.append({
                "type": "mcq",
                "domain": node.get("domain", "1.0 General Security Concepts"),
                "sub_objective": sub_obj,
                "question": f"Regarding **{sub_obj}**, which technical implementation applies official guidelines?",
                "options": ["Enforcing strict cryptographic controls", "Disabling audit logging", "Using legacy protocols", "Permitting implicit trust"],
                "correct_index": 0
            })

    # Ensure PBQs are strictly front-loaded, and cap the pool at exactly 45 items
    pbqs = [q for q in pool if q.get("type") == "pbq"]
    others = [q for q in pool if q.get("type") != "pbq"]
    _CACHED_EXAM_POOL = (pbqs + others)[:45]
    return _CACHED_EXAM_POOL


# --- LangGraph Socratic FSM Graph ---
def socratic_tutor(state: ChatState) -> dict:
    sub_topic = state.get("sub_topic", COURSE_TITLE)
    node_id = state.get("node_id", "")
    system_instruction = get_system_context()
    node = get_node_by_id(node_id)

    directive = f"""[SYSTEM DIRECTIVE]
Target Revision Objective: {sub_topic}
Required Acronyms: {', '.join(node.get('required_acronyms', []))}
Core Terms: {', '.join(node.get('key_terms', []))}

TUTORING MANDATE:
- Guide the candidate step-by-step using probing questions and constructive hints.
- Never give away full answers directly. Demand precise CompTIA terminology.
"""

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
        max_output_tokens=2048
    )

    contents = convert_messages_to_contents(state["messages"])
    contents.insert(0, types.Content(role="user", parts=[types.Part.from_text(text=directive)]))

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=config
    )

    return {"messages": state["messages"] + [HumanMessage(content=response.text)]}


def didactic_fallback(state: ChatState) -> dict:
    sub_topic = state.get("sub_topic", "")
    system_instruction = get_system_context()

    directive = f"""[SYSTEM DIRECTIVE]
You are a Lead CompTIA Security+ ({EXAM_CODE}) Senior Examiner concluding a Socratic session.
Topic Focus: {sub_topic}

FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS:

[Detailed technical validation of the candidate's responses and corrected architecture solution]

### Dialogue Phase Complete
- **Targeted Feedback:** 
  - [Constructive feedback point 1]
  - [Constructive feedback point 2]
- **CompTIA Security+ Terminology:**
  - **Keywords Used Well:** [Term 1, Term 2]
  - **Missed Terms to Master:** [Term 3, Term 4]
"""

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
        max_output_tokens=2048
    )

    contents = convert_messages_to_contents(state["messages"])
    contents.insert(0, types.Content(role="user", parts=[types.Part.from_text(text=directive)]))

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=config
    )

    return {"messages": state["messages"] + [HumanMessage(content=response.text)]}


def route_turn(state: ChatState) -> str:
    if state.get("is_validated", False):
        return "didactic_fallback"
    return "socratic_tutor"


builder = StateGraph(ChatState)
builder.add_node("socratic_tutor", socratic_tutor)
builder.add_node("didactic_fallback", didactic_fallback)

builder.set_conditional_entry_point(
    route_turn,
    {
        "socratic_tutor": "socratic_tutor",
        "didactic_fallback": "didactic_fallback",
    },
)

builder.add_edge("socratic_tutor", END)
builder.add_edge("didactic_fallback", END)

workflow = builder.compile()


# --- Synthesis Session Exit Helpers ---
def generate_socratic_synthesis_prompt(sub_topic: str, chat_history: list) -> dict:
    """Generates a final synthesis summary exercise upon exiting the uncapped dialogue."""
    transcript_lines = []
    for msg in chat_history:
        role = "Candidate" if msg.get("role") == "student" or isinstance(msg, HumanMessage) else "Engine"
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        transcript_lines.append(f"{role}: {content}")
    full_transcript = "\n".join(transcript_lines)

    system_instruction = get_system_context()
    prompt = f"""
    Target Subtopic: {sub_topic}
    TRANSCRIPT:
    {full_transcript}

    Create a 6-point synthesis task summarizing the key technical mechanisms discussed.

    Return ONLY a JSON object:
    {{
        "question": "Construct a concise technical synthesis summarizing the core concepts and precise controls explored in today's session on '{sub_topic}'.",
        "mark_scheme": [
            "Point 1: [Key mechanism 1]",
            "Point 2: [Key mechanism 2]",
            "Point 3: [Key mechanism 3]",
            "Point 4: [Point 4]",
            "Point 5: [Point 5]",
            "Point 6: [Technical conclusion]"
        ]
    }}
    """

    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2, max_output_tokens=2048)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)

    clean_text = re.sub(r"```json|```", "", extract_clean_text(response)).strip()
    return json.loads(clean_text)


def evaluate_socratic_synthesis(question: str, mark_scheme: list, student_answer: str, chat_history: list) -> dict:
    """Evaluates final synthesis answer."""
    ms_text = "\n".join([f"- Point {i+1}: {pt}" for i, pt in enumerate(mark_scheme)])
    
    system_instruction = get_system_context()
    prompt = f"""
    Task Prompt: {question}
    Rubric:
    {ms_text}

    Candidate Response:
    {student_answer}

    Return ONLY JSON:
    {{
        "score": 5,
        "max_score": 6,
        "awarded_points": ["Point awarded..."],
        "missed_points": ["Point missed..."],
        "keywords_used": ["AES-256"],
        "keywords_missed": ["SHA-256"],
        "feedback": "Examiner evaluation..."
    }}
    """

    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0, max_output_tokens=2048)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)

    clean_text = re.sub(r"```json|```", "", extract_clean_text(response)).strip()
    return json.loads(clean_text)