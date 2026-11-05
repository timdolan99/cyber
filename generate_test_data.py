import os
import random
import json
from datetime import datetime, timedelta
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# --- 1. Database Connection ---
try:
    DB_URL = st.secrets["postgres"]["url"]
except Exception:
    DB_URL = os.getenv("POSTGRES_URL", os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/postgres"))

engine = create_engine(DB_URL, pool_pre_ping=True, echo=False)
Base = declarative_base()

# --- 2. Matching CompTIA Security+ ORM Schema ---
class ActivityLog(Base):
    __tablename__ = "activity_logs_comptia_csp"

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

APP_MODES = ["socratic_drill", "response_refactor", "pbq_tester", "exam_mode"]

# --- 3. Security+ Acronym & Keyword Map ---
SECURITY_KEYWORD_MAP = {
    "1.1 Compare and contrast various security control types": {
        "keywords": ["Preventive", "Detective", "Corrective", "Deterrent", "Compensating", "Physical", "Technical", "Administrative"],
    },
    "1.2 Summarize fundamental security concepts": {
        "keywords": ["CIA Triad", "Confidentiality", "Integrity", "Availability", "Non-repudiation", "Zero Trust", "Least Privilege", "Defense-in-Depth"],
    },
    "2.1 Compare and contrast threat actor types and intelligence sources": {
        "keywords": ["APT", "Insider Threat", "Script Kiddie", "Hacktivist", "OSINT", "Threat Intelligence", "STIX", "TAXII"],
    },
    "2.2 Explain vulnerability scanning and penetration testing concepts": {
        "keywords": ["CVSS", "False Positive", "Credentialed Scan", "Black Box", "White Box", "Gray Box", "Rules of Engagement", "Penetration Testing"],
    },
    "3.1 Implement secure network architecture concepts": {
        "keywords": ["DMZ", "VLAN", "Subnetting", "NAT", "VPN", "Zero Trust Network Architecture", "Microsegmentation", "WAF"],
    },
    "3.4 Summarize secure cryptographic solutions": {
        "keywords": ["AES", "RSA", "SHA-256", "PKI", "Salting", "Hashing", "Symmetric", "Asymmetric", "Digital Signature"],
    },
    "4.1 Implement identity and access management": {
        "keywords": ["IAM", "MFA", "SSO", "RBAC", "ABAC", "LDAP", "Kerberos", "OAuth", "SAML", "Federation"],
    },
    "4.2 Explain basic incident response processes": {
        "keywords": ["Containment", "Eradication", "Lessons Learned", "Chain of Custody", "Forensic Image", "Playbooks", "SIEM", "EDR"],
    },
    "5.1 Summarize elements of governance, risk, and compliance": {
        "keywords": ["GDPR", "HIPAA", "PCI-DSS", "Risk Tolerance", "Risk Mitigation", "Business Impact Analysis", "RTO", "RPO"],
    }
}

DEFAULT_KEYWORDS = ["CIA Triad", "Access Control", "Risk Assessment", "Mitigation Strategy"]

# --- 4. Generation Logic ---
def generate_synthetic_data(num_records=1800):
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    spec_path = "comptia_secplus_spec.json"
    if os.path.exists(spec_path):
        with open(spec_path, "r", encoding="utf-8") as f:
            course_spec = json.load(f)
    else:
        print("Error: comptia_secplus_spec.json not found.")
        return

    subject_title = course_spec.get("course_title", "CompTIA Security+")
    exam_code = course_spec.get("code", "SY0-701")
    full_subject_name = f"{subject_title} ({exam_code})"
    course_level = course_spec.get("target_audience", "Adult Professional")

    domains_dict = course_spec.get("domains", {})
    if not domains_dict:
        print("Error: comptia_secplus_spec.json has no domains defined.")
        return

    topic_pool = []
    for domain_name, subtopics in domains_dict.items():
        for subtopic in subtopics:
            topic_pool.append((domain_name, subtopic))

    if not topic_pool:
        print("Error: No sub-objectives extracted from specification.")
        return

    print(f"Seeding {num_records} activity logs for '{full_subject_name}' into table 'activity_logs_comptia_csp'...")

    start_date = datetime.utcnow() - timedelta(days=30)

    for _ in range(num_records):
        domain_name, subtopic_name = random.choice(topic_pool)

        domain_meta = SECURITY_KEYWORD_MAP.get(subtopic_name, {"keywords": DEFAULT_KEYWORDS})
        app_mode = random.choice(APP_MODES)

        base_score = random.choices([40.0, 55.0, 70.0, 85.0, 100.0], weights=[0.1, 0.25, 0.35, 0.2, 0.1])[0]
        score_pct = max(10.0, min(100.0, base_score + random.uniform(-10.0, 10.0)))

        all_kw = list(domain_meta["keywords"])
        num_used = int((score_pct / 100.0) * len(all_kw))
        random.shuffle(all_kw)
        used_kw = all_kw[:num_used]
        missed_kw = all_kw[num_used:]

        misconception_flag = 1 if score_pct < 50.0 else 0

        timestamp = start_date + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(8, 22),
            minutes=random.randint(0, 59)
        )

        log = ActivityLog(
            timestamp=timestamp,
            subject=full_subject_name,
            level=course_level,
            unit=domain_name,
            subtopic=subtopic_name,
            app_mode=app_mode,
            score_pct=round(score_pct, 1),
            keywords_used=", ".join(used_kw) if used_kw else None,
            keywords_missed=", ".join(missed_kw) if missed_kw else None,
            misconception_flag=misconception_flag
        )

        session.add(log)

    session.commit()
    session.close()
    print(f"Successfully populated {num_records} activity logs into 'activity_logs_comptia_csp'!")

if __name__ == "__main__":
    generate_synthetic_data(1)