
import json
import os
import sys
from pathlib import Path
from pypdf import PdfReader
from google import genai
from google.genai import types

# Use built-in tomllib for Python 3.11+ or fallback to tomli/custom parsing
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

def load_api_key() -> str:
    """Load API key from .streamlit/secrets.toml, secrets directory, or environment variables."""
    base_dir = Path(__file__).resolve().parent
    
    # Check .streamlit/secrets.toml first
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

    # Fallback to system environment variables
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Could not locate a valid Gemini API key in .streamlit/secrets.toml or environment variables."
        )
    return key

def extract_pdf_text(pdf_path):
    """Reads all text from the local syllabus PDF."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find syllabus PDF at {path.resolve()}")
    
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    return full_text

def generate_node_definitions():
    base_dir = Path(__file__).resolve().parent
    # Absolute path resolve to syllabus\cs_plus_701_objectives.pdf
    pdf_path = base_dir / "syllabus" / "cs_plus_701_objectives.pdf"

    print(f"📖 Extracting syllabus text from {pdf_path.resolve()}...")
    pdf_content = extract_pdf_text(pdf_path)
    print(f"✅ Extracted {len(pdf_content)} characters.")

    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert CompTIA Security+ item developer and curriculum engineer.
    Below is the text extracted from the official Security+ (SY0-701) syllabus PDF:

    ---
    {pdf_content[:25000]}
    ---

    Using this corpus as your source of ground truth, generate a detailed FSM node repository for key CompTIA sub-objectives across all domains.
    
    Return ONLY a raw JSON array matching this exact schema for each objective:
    [
      {{
        "node_id": "1.2_fundamental_security_concepts",
        "domain": "1.0 General Security Concepts",
        "sub_objective": "1.2 Summarize fundamental security concepts (CIA triad, AAA, Zero Trust)",
        "key_terms": ["Confidentiality", "Integrity", "Availability", "Authentication", "Authorization", "Accounting", "Zero Trust"],
        "required_acronyms": ["CIA", "AAA", "MFA", "PoLP"],
        "common_misconceptions": [
          "Confusing Hashing (Integrity) with Encryption (Confidentiality)",
          "Assuming Zero Trust means no internal authentication is required"
        ],
        "refactor_sample_flawed_text": "To protect our data in transit, we hash it with AES so that unauthorized people cannot read the contents.",
        "refactor_sample_corrected_text": "To protect data in transit, we encrypt it using symmetric ciphers like AES to ensure Confidentiality. Hashing (e.g., SHA-256) is used to verify Integrity, not to prevent unauthorized reading.",
        "pbq_sample_log_snippet": "192.168.1.15 -> 10.0.0.5 TCP 443 [SYN]\\n192.168.1.15 -> 10.0.0.5 TCP 80 [SYN]\\n192.168.1.15 -> 10.0.0.5 TCP 22 [SYN]"
      }}
    ]

    Do NOT include markdown fences, HTML, or conversational commentary.
    """

    print("🤖 Processing node metadata with Gemini...")
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )

    raw_text = response.text.strip()
    clean_json = raw_text.replace("```json", "").replace("```", "").strip()

    nodes_data = json.loads(clean_json)

    output_path = base_dir / "comptia_nodes.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nodes_data, f, indent=2)

    print(f"🎉 Successfully generated {output_path.name} with {len(nodes_data)} FSM state nodes!")

if __name__ == "__main__":
    generate_node_definitions()