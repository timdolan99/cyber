import json
import os
import sys
from pathlib import Path
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
                        # Check standard Streamlit key formats
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
                # Manual fallback line parsing if tomllib is unavailable
                for line in secret_path.read_text(encoding="utf-8").splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.split("=", 1)
                        if k.strip().upper() in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY"]:
                            return v.strip().strip('"').strip("'")

    # Check additional secrets directory paths
    possible_secret_paths = [
        base_dir / "secrets" / "gemini_api_key.txt",
        base_dir / "secrets" / "api_key.txt",
        base_dir / "secrets" / "secrets.json",
        base_dir.parent / "secrets" / "gemini_api_key.txt"
    ]

    for secret_path in possible_secret_paths:
        if secret_path.is_file():
            if secret_path.suffix == ".json":
                try:
                    with open(secret_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        key = data.get("GEMINI_API_KEY") or data.get("api_key") or data.get("GOOGLE_API_KEY")
                        if key:
                            return key.strip()
                except Exception:
                    continue
            else:
                key = secret_path.read_text(encoding="utf-8").strip()
                if key:
                    return key

    # Fallback to system environment variables
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Could not locate a valid Gemini API key in .streamlit/secrets.toml, "
            "the secrets folder, or environment variables."
        )
    return key

# Initialize Client
api_key = load_api_key()
client = genai.Client(api_key=api_key)

def generate_comptia_spec(output_path="comptia_secplus_spec.json"):
    prompt = """
    You are an expert cybersecurity curriculum designer and CompTIA Security+ item developer.
    Generate a complete, structured JSON specification file for the CompTIA Security+ exam based on its official domains.
    
    Return ONLY a raw JSON object matching this exact structure:
    {
      "course_title": "CompTIA Security+",
      "code": "SY0-701",
      "target_audience": "Adult Professional / Cyber Specialist",
      "socratic_mode": "Uncapped",
      "domains": {
        "1.0 General Security Concepts": [
          "1.1 Compare and contrast security control categories and types",
          "1.2 Summarize fundamental security concepts (CIA triad, AAA, Zero Trust)",
          "1.3 Explain the importance of change management processes",
          "1.4 Explain basic cryptographic concepts (Public key infrastructure, hashing, digital signatures)"
        ],
        "2.0 Threats, Vulnerabilities, and Mitigations": [
          "2.1 Compare and contrast common threat actor types and attributes",
          "2.2 Explain common threat vectors and attack surfaces",
          "2.3 Analyze indicators of malicious activity (Social engineering, malware, application attacks)",
          "2.4 Analyze mitigation techniques to secure environments"
        ],
        "3.0 Security Architecture": [
          "3.1 Compare and contrast security architecture models (Cloud, Hybrid, On-premise)",
          "3.2 Summarize secure infrastructure design principles (Segmentation, DMZ, Air gaps)",
          "3.3 Explain data protection strategies"
        ],
        "4.0 Security Operations": [
          "4.1 Apply common security baseline controls on host systems",
          "4.2 Analyze logs and data sources during security investigations",
          "4.3 Explain incident response processes and digital forensics fundamentals"
        ],
        "5.0 Security Program Management and Oversight": [
          "5.1 Summarize elements of effective security governance",
          "5.2 Explain risk management processes and assessment methods",
          "5.3 Summarize compliance requirements and privacy standards"
        ]
      }
    }
    Do NOT include markdown fences, HTML, or conversational commentary.
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )

    raw_text = response.text.strip()
    clean_json = raw_text.replace("```json", "").replace("```", "").strip()

    spec_data = json.loads(clean_json)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec_data, f, indent=2)

    print(f"✅ Successfully generated {output_path}!")

if __name__ == "__main__":
    generate_comptia_spec()