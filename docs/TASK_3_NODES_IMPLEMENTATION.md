# Task 3 Implementation Summary

## Overview
Implemented all six agent nodes as separate modules under `app/agent/nodes/` with mock implementations for testing without Devang's NLP or Navya's ML classifier.

## Files Created

### Core Node Files (app/agent/nodes/)
- **language_detect.py** — Language detection from first message, initial symptom extraction
- **symptom_collector.py** — Multi-turn symptom collection with follow-up questions (max 5)
- **classifier.py** — Disease classification and urgency prediction
- **rag_advisor.py** — Medical advice retrieval (with FAISS placeholder)
- **response_composer.py** — Format final response in patient's language
- **escalation.py** — Handle emergency/urgent case escalation and PHC lookup

### Mock/Integration Files
- **app/nlp/models.py** — Updated with `process_message()`, `translate_to_en()`, `translate_to_patient_lang()` functions with MOCK_NLP support
- **app/ml/classifier.py** — Updated with `predict_triage()` function with MOCK_CLASSIFIER support
- **app/db/crud.py** — Added `get_nearest_phc()` stub for PHC lookup
- **app/agent/graph.py** — Updated to import and use individual node functions

### Testing & Documentation
- **app/agent/TESTING.md** — Comprehensive guide for testing nodes with mock implementations

## Environment Variables

Use these to toggle mock implementations:

```bash
# PowerShell
$env:MOCK_NLP='true'
$env:MOCK_CLASSIFIER='true'

# Bash
export MOCK_NLP=true
export MOCK_CLASSIFIER=true
```

## Mock Behavior

### MOCK_NLP=true
- `detect_language()` → always returns "hi"
- `process_message()` → returns `["fever", "cough"]`
- `translate_to_en()` → returns text as-is
- `translate_to_patient_lang()` → appends language code to simulate translation

### MOCK_CLASSIFIER=true
- `predict_triage()` → returns hardcoded triage result:
  - urgency: "routine"
  - conditions: ["Common Cold", "Upper Respiratory Infection", "Influenza"]

## Graph Architecture

```
START
  ↓
language_detect (detect language, extract symptoms)
  ↓
symptom_collector ⟲ (loop: ask Q until 3+ symptoms OR 5 Q's OR patient says "bas")
  ↓
classifier (predict urgency + top-3 conditions)
  ↓
rag_advisor (retrieve medical advice)
  ↓
escalation (find nearest PHC if urgent/emergency)
  ↓
response_composer (format final message in patient's language)
  ↓
END
```

## Node Return Values

All nodes return **partial state updates** (dicts), not full state objects. This follows LangGraph best practices:

- `language_detect_node()` → `{"language": str, "symptoms": list, "medical_entities": dict, "question_count": int}`
- `symptom_collector_node()` → `{}` | `{"messages": list, "question_count": int}`
- `classifier_node()` → `{"urgency": str, "conditions": list}`
- `rag_advisor_node()` → `{"advice": str}`
- `escalation_node()` → `{}` | `{"nearest_phc": dict}`
- `response_composer_node()` → `{"messages": list}`

## Testing the Complete Flow

```python
import os
from app.agent.graph import triage_graph
from app.agent.states import TriageState

os.environ["MOCK_NLP"] = "true"
os.environ["MOCK_CLASSIFIER"] = "true"

state = TriageState(
    patient_phone="919876543210",
    language="en",
    messages=[{"role": "user", "content": "I have fever and cough", "language": "en"}],
    symptoms=[],
    medical_entities={},
    question_count=0,
    urgency="",
    conditions=[],
    advice="",
    nearest_phc={},
    follow_up_scheduled=False,
)

output = triage_graph.invoke(state)
print(output["messages"][-1]["content"])  # Final response
```

## Next Steps

1. **Integrate Devang's NLP models:**
   - Replace mock `process_message()` with actual MuRIL BERT NER
   - Replace mock `translate_to_en()` with IndicTrans2
   - Replace mock `translate_to_patient_lang()` with IndicTrans2

2. **Integrate Navya's ML classifier:**
   - Replace mock `predict_triage()` with actual XGBoost model

3. **Implement FAISS RAG:**
   - Populate medical knowledge base
   - Implement actual `retrieve_medical_advice()` with vector search

4. **Doctor alerts:**
   - Implement doctor notification via WhatsApp/SMS in `escalation_node()`
   - Create background task for late responses

5. **Integration tests:**
   - Test end-to-end flow with actual models
   - Validate symptom collection loop
   - Test escalation routing

## File Structure

```
app/agent/
├── __init__.py
├── states.py              # TriageState TypedDict
├── graph.py               # Graph builder (imports nodes)
├── TESTING.md             # Testing guide
├── nodes/
│   ├── __init__.py
│   ├── language_detect.py
│   ├── symptom_collector.py
│   ├── classifier.py
│   ├── rag_advisor.py
│   ├── response_composer.py
│   └── escalation.py
```

## Conditional Edges

### symptom_collector → ? (should_continue_collecting_symptoms)
- **→ classifier** if: `len(symptoms) >= 3` OR `question_count >= 5` OR patient says "bas"/"that's all"
- **→ symptom_collector** otherwise

### escalation → ? (should_escalate)
- **→ response_composer** (always)

## Known Limitations (To Be Addressed)

1. **PHC lookup** — Currently returns mock data. Needs geolocation + database integration.
2. **FAISS integration** — Not yet connected. Using template advice for now.
3. **Doctor alerts** — Not yet implemented. Needs background task + WhatsApp/SMS API.
4. **Multi-turn handling** — Graph currently expects all input upfront. Webhook needs to persist state between requests.
5. **Location sharing** — No support yet for patient location. Needed for accurate PHC distance.

## Configuration for Production

When deploying with real models, ensure:

1. NLP models are pre-loaded at startup (via FastAPI lifespan)
2. XGBoost model is loaded and cached
3. FAISS index is populated and loaded
4. PHC database is connected
5. Environment variables are properly set (MOCK_NLP=false, MOCK_CLASSIFIER=false)
