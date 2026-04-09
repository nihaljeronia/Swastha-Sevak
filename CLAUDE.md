# CLAUDE.md — Complete Project Context for Claude Code

> This file contains everything Claude Code needs to know about the Swasthya Sevak project.
> Read this FULLY before making any changes to the codebase.

---

## 1. PROJECT IDENTITY

**Name:** Swasthya Sevak (स्वास्थ्य सेवक — "Health Servant")
**What it is:** An Agentic AI medical triage chatbot for rural India, delivered entirely via WhatsApp.
**Who it's for:** Rural patients who message the bot directly from their own phones — NO intermediary (no ASHA worker, no health worker operating the system). The patient IS the user.
**Built by:** A 3rd year BTech CSE (AI/ML) student as a project for PS-03: "Agentic AI for Rural Healthcare Triage"

---

## 2. THE PROBLEM

- In rural India, doctor-to-patient ratio is critically low
- Patients travel hours for conditions that could be managed locally
- Or worse — they ignore life-threatening symptoms because there's no doctor nearby
- ~85.5% of Indian households have a smartphone, ~97% of internet users have WhatsApp
- WhatsApp works on 2G/3G, patients already know how to use it

## 3. THE SOLUTION

Patients message a WhatsApp number with text or voice notes in ANY Indian language. The AI bot:
1. Auto-detects their language (no menu, no "Press 1 for Hindi")
2. Collects symptoms conversationally (3-5 targeted questions)
3. Classifies urgency using a trained ML classifier (NOT generative LLM guessing)
4. Sends actionable advice: self-care instructions / nearest PHC name+phone+distance / emergency 108 info
5. Alerts the nearest doctor directly if emergency
6. Follows up 24-48 hours later: "How are you feeling now?"

---

## 4. WHY NOT JUST USE CHATGPT/GEMINI?

This is a critical differentiator. General LLMs fail at medical triage:
- Nature Medicine (Feb 2026): ChatGPT Health undertriaged 52% of emergencies
- BMJ study: only 56% overall triage accuracy — coin-flip level
- They can't process WhatsApp voice notes
- Weak on Indian languages beyond Hindi text (no Bhojpuri, Chhattisgarhi, code-switching)
- No structured clinical protocol — different answer every time
- No local disease context (dengue season in MP, malaria belts)
- Output is "consult a doctor" not "PHC Raisen Road, 8km, phone: 07482-XXXXXX"
- No follow-up, no doctor alerts, no patient tracking
- Data goes to US servers — no DPDP Act compliance
- ChatGPT Plus costs ₹1,650/month — our bot is free

Our system uses protocol-driven classification (XGBoost), not generative guessing.

---

## 5. ARCHITECTURE OVERVIEW

```
Patient (WhatsApp) 
    → Meta Cloud API (webhook) 
    → FastAPI backend 
    → LangGraph agentic engine
    → AI4Bharat NLP models (IndicWhisper, MuRIL, IndicTrans2)
    → XGBoost disease classifier
    → FAISS RAG for medical advice
    → Reply sent back via Meta API
    → Follow-up scheduled via BackgroundTasks
    → Emergency: doctor alerted via WhatsApp/SMS
```

### Three layers:
1. **WhatsApp Integration Layer** — Meta Cloud API webhooks, message parsing, reply sending
2. **Agentic Backend** — FastAPI + LangGraph state machine, RAG, classification
3. **AI/NLP Layer** — IndicWhisper (ASR), MuRIL BERT (NLU/NER), IndicTrans2 (translation), XGBoost (classifier)

### Supporting systems:
- PostgreSQL — patient records, triage sessions, symptom logs, alerts
- FAISS — vector store for medical knowledge RAG
- FastAPI BackgroundTasks — follow-ups and doctor alerts (NOT Celery, NOT Redis)
- Next.js admin dashboard — for doctors/PHC staff (build later)

---

## 6. TECH STACK (FINAL DECISIONS)

| Component | Technology | Why |
|---|---|---|
| Backend framework | FastAPI (async) | Webhook-driven, async, Python ecosystem |
| Agent framework | LangGraph | Stateful triage loop with checkpointing |
| RAG framework | LangChain | Vector retrieval from medical KB |
| WhatsApp API | Meta Cloud API (free tier) | Direct, no Twilio, 1000 free conversations/month |
| Database | PostgreSQL + SQLAlchemy 2.0 async + asyncpg | JSON columns for symptoms, production-grade |
| Migrations | Alembic | Schema versioning |
| Speech-to-text | AI4Bharat IndicWhisper | 22 Indian languages, trained on Indian accents |
| Language understanding | Google MuRIL BERT | 17 Indian languages + transliterated variants |
| Translation | AI4Bharat IndicTrans2 | 22 scheduled Indian languages, open-source |
| Text-to-speech | AI4Bharat Indic Parler-TTS | Voice replies in patient's language |
| Disease classifier | XGBoost / Random Forest | Deterministic, auditable, trained on symptom data |
| Vector store | FAISS (faiss-cpu) | Medical knowledge RAG |
| Embeddings | MuRIL / sentence-transformers | Encode medical text for retrieval |
| Background tasks | FastAPI BackgroundTasks | Follow-ups, alerts (NOT Celery) |
| Tunneling (dev) | VS Code Port Forwarding | Exposes localhost to Meta webhook |
| Validation | Pydantic v2 | All schemas and settings |
| Testing | pytest + pytest-asyncio | Async test support |

### What we explicitly DO NOT use:
- **Docker** — not needed for development, adds complexity
- **Celery + Redis** — overkill for project scope, BackgroundTasks is sufficient
- **SQLite** — student wanted PostgreSQL (correct choice for JSON columns and production)
- **Twilio** — unnecessary cost, Meta Cloud API is free and direct
- **ngrok** — using VS Code port forwarding instead

---

## 7. DATABASE SCHEMA

### Patient
- id: UUID, primary key
- phone: String, unique, indexed (WhatsApp number with country code)
- name: String, nullable (collected later if patient shares it)
- language: String, nullable (auto-detected from first message, e.g., "hi", "ta", "mr")
- district: String, nullable
- state: String, nullable
- created_at: DateTime
- updated_at: DateTime

### TriageSession
- id: UUID, primary key
- patient_id: FK → Patient
- symptoms: JSON (list of extracted symptom strings)
- medical_entities: JSON (dict of NER output: body_parts, duration, severity)
- urgency: String (one of: "self_care", "routine", "urgent", "emergency")
- conditions: JSON (list of top-3 predicted conditions with confidence)
- advice: Text (the advice text sent to patient)
- follow_up_at: DateTime, nullable (when to send follow-up)
- follow_up_sent: Boolean, default False
- status: String (one of: "in_progress", "completed", "escalated")
- created_at: DateTime
- completed_at: DateTime, nullable

### Message
- id: UUID, primary key
- session_id: FK → TriageSession
- patient_id: FK → Patient
- direction: String ("inbound" or "outbound")
- message_type: String ("text", "audio", "interactive")
- content: Text (message body or transcription)
- raw_payload: JSON (full Meta webhook payload for debugging)
- created_at: DateTime

### Alert
- id: UUID, primary key
- session_id: FK → TriageSession
- doctor_phone: String
- alert_type: String ("emergency", "urgent_referral")
- patient_summary: Text (structured summary sent to doctor)
- status: String ("pending", "sent", "acknowledged")
- sent_at: DateTime, nullable
- created_at: DateTime

---

## 8. LANGGRAPH AGENT FLOW

The triage agent is a LangGraph StateGraph with these nodes:

```
START
  ↓
language_detect → detect language from first message (MuRIL / fastText)
  ↓
symptom_collector → ask follow-up questions, extract symptoms via NER
  ↓ (loops back if < 3 symptoms collected, max 5 questions)
  ↓ (proceeds if enough symptoms OR patient says "that's all")
classifier → XGBoost predicts urgency + top conditions from symptom vector
  ↓
rag_advisor → retrieve relevant medical advice from FAISS knowledge base
  ↓
response_composer → compose reply in patient's language using IndicTrans2
  ↓
escalation → if emergency/urgent: alert nearest doctor, send 108 info
  ↓
END (schedule follow-up via BackgroundTasks)
```

### Agent State (TypedDict):
```python
class TriageState(TypedDict):
    patient_phone: str
    language: str                    # detected language code
    messages: list[dict]             # conversation history
    symptoms: list[str]             # extracted symptoms in English
    medical_entities: dict          # NER output
    question_count: int             # how many Qs asked so far
    urgency: str                    # classifier output
    conditions: list[dict]          # top-3 with confidence scores
    advice: str                     # RAG-generated advice
    nearest_phc: dict               # name, distance, phone
    follow_up_scheduled: bool
```

---

## 9. WHATSAPP CONVERSATION DESIGN

### Principles:
- Voice-first — treat voice notes as primary input, text as secondary
- No menus on first message — patient just speaks/types naturally
- Numbered options for follow-up questions (patient types "1", "2", etc.)
- Short sentences, no medical jargon
- Language of reply matches language patient used
- Bot speaks like a helpful neighbor, not a clinical system

### Example conversation (Marathi):
```
Patient: 🎤 [voice: "mala 3 divsapasun tap ahe ani angavar puravya aalya"]

Bot: नमस्कार! मी तुमच्या आरोग्यासाठी मदत करतो.
     तुम्हाला 3 दिवसांपासून ताप आणि पुरळ आले.
     
     तुमचं वय किती आहे?

Patient: 34

Bot: तुम्हाला यापैकी काही त्रास होत आहे का?
     1️⃣ डोकेदुखी
     2️⃣ सांधेदुखी
     3️⃣ उलटी
     4️⃣ नाकातून रक्त
     5️⃣ यातलं काहीच नाही

Patient: 2

Bot: ⚠️ तुमची लक्षणे डेंग्यूसारखी दिसत आहेत.
     🔴 कृपया लवकरात लवकर जवळच्या दवाखान्यात जा.
     📍 जवळचा: ग्रामीण रुग्णालय, भोपाळ रोड (8 km)
     📞 फोन: 07482-XXXXXX
     💧 भरपूर पाणी प्या. ॲस्पिरिन घेऊ नका.

[24 hours later]
Bot: नमस्कार! काल तुम्हाला ताप आणि पुरळ होतं. आता कसं वाटतंय?
     1️⃣ बरं वाटतंय
     2️⃣ तसंच आहे
     3️⃣ अजून वाईट झालंय
```

---

## 10. CURRENT PROJECT STATUS

### ✅ COMPLETED:
- FastAPI scaffold with async startup lifespan
- app/core/config.py — Pydantic settings with env vars
- app/db/ — SQLAlchemy async session, models stubs, CRUD stubs
- app/webhook/ — routes, schemas, service layer
- app/nlp/, app/ml/, app/agents/ — stub modules
- app/tasks/ — followup.py and alerts.py as async BackgroundTasks stubs
- Async pytest scaffolding in tests/
- Meta WhatsApp Cloud API sandbox working
- GET /webhook — Meta verification ✅ (returns hub.challenge)
- POST /webhook — Receives messages, parses text/audio/interactive, sends echo reply ✅
- VS Code port forwarding working as tunnel
- Removed Docker, Celery, Redis from scaffold
- **Step 2: Database layer** ✅
  - 4 SQLAlchemy 2.0 models: Patient, TriageSession, Message, Alert (UUID PKs, JSON columns, relationships)
  - Alembic async migrations configured and applied
  - Full CRUD layer: get_or_create_patient, save_message, triage session management, alerts
  - Service layer bridging webhook → DB (routes never import db directly)
  - Webhook routes updated: BackgroundTasks, persists inbound + outbound messages
  - All tables verified in PostgreSQL

### ⬜ NEXT (in order):
1. **Step 3: NLP pipeline** — IndicWhisper ASR, language detection, MuRIL NER, IndicTrans2
2. **Step 4: Triage agent** — LangGraph state machine with symptom collection loop
3. **Step 5: Response + follow-up** — Compose multilingual reply, schedule follow-up, doctor alerts

---

## 11. CODING RULES FOR CLAUDE CODE

### Architecture:
- Webhook layer (app/webhook/) must NEVER import from app/db/ directly — go through a service layer
- All FastAPI handlers must be async — NEVER use synchronous DB or HTTP calls
- All NLP model loading happens at startup via FastAPI lifespan, not per-request
- Agent state is managed by LangGraph checkpoints, not manual session tracking
- Webhook POST returns 200 immediately, processes message via BackgroundTasks
- All medical decisions flow through XGBoost classifier, NEVER through raw LLM generation
- Translation happens at input (to English) and output (to patient language), nowhere else

### Code style:
- snake_case for variables/functions, PascalCase for classes
- Type hints on ALL functions and return types
- Pydantic v2 models for all schemas
- Docstrings on every module, class, and public function
- No inline SQL — SQLAlchemy ORM only
- Secrets via Pydantic Settings (.env), never hardcoded
- All external HTTP calls via httpx.AsyncClient, not requests library
- Use `from __future__ import annotations` for forward references

### File conventions:
- Schemas → schemas.py within each module
- Database models → app/db/models.py ONLY
- CRUD operations → app/db/crud.py ONLY
- Background tasks → app/tasks/ ONLY
- NLP model wrappers → app/nlp/ ONLY
- ML classifier → app/ml/ ONLY
- Agent nodes → app/agent/nodes/ ONLY

### Testing:
- pytest with pytest-asyncio
- Fixtures in tests/conftest.py
- Mock all external services (Meta API, NLP models) in tests
- Test each agent node independently before full graph
- Never make real API calls in tests

---

## 12. ENVIRONMENT VARIABLES (.env)

```env
# Meta WhatsApp Cloud API
META_VERIFY_TOKEN=swastha-sevak-token
META_ACCESS_TOKEN=EAAxxxxxxx...  # generate from Meta Business Settings → System Users
META_PHONE_NUMBER_ID=123456789012345

# Database
DATABASE_URL=postgresql+asyncpg://postgres:yourpassword@127.0.0.1:5432/swasthya_sahayak

# Model paths (will be set when we download models)
INDICWHISPER_MODEL=ai4bharat/indicwhisper-base
MURIL_MODEL=google/muril-base-cased
INDICTRANS2_MODEL=ai4bharat/indictrans2-en-indic-1B
FAISS_INDEX_PATH=./data/faiss_index
CLASSIFIER_MODEL_PATH=./app/ml/models/triage_classifier.joblib
```

---

## 13. KEY DATASETS

- **Symptom-disease mapping:** Kaggle disease-symptom datasets (~400 diseases), ICMR guidelines
- **Medical knowledge base:** WHO guidelines, ICMR protocols (PDFs ingested into FAISS)
- **PHC directory:** Name, district, state, lat/lon, phone for primary health centers
- **MedIC dataset:** AIIMS medical query dataset for Indian languages
- **Training data for NER:** Fine-tune MuRIL on medical entity recognition

---

## 14. DEPLOYMENT PLAN (later)

- Backend: Railway or Render (free tier for PostgreSQL + FastAPI)
- Model hosting: HuggingFace Inference API or self-hosted on a GPU VPS
- Permanent WhatsApp number: Meta business verification (need GST/MSME certificate)
- Admin dashboard: Vercel (Next.js)

---

## 15. IMPORTANT: WHAT TO BUILD NEXT

Step 2 (Database layer) is **COMPLETE**.

The immediate task is **Step 3: NLP Pipeline**. This means:

1. Integrate AI4Bharat IndicWhisper for speech-to-text (voice notes → English text)
2. Implement language detection (fastText or MuRIL-based)
3. Set up MuRIL BERT for medical NER (extract symptoms, body parts, duration, severity)
4. Integrate IndicTrans2 for translation (patient language → English, English → patient language)
5. Wire the NLP pipeline into the webhook flow

After that: Step 4 (LangGraph agent), Step 5 (response + follow-up).
