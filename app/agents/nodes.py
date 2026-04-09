from typing import Any
import asyncio
import logging
from dotenv import load_dotenv
import os

load_dotenv()

from app.agents.state import TriageState
from app.nlp.models import NLPModelManager
from app.ml.classifier import DiseaseClassifier

logger = logging.getLogger(__name__)

# Note: In production we'd pass these as dependencies or get from app state
# For now, we instantiate them if they aren't provided, or they should be provided via graph config/metadata.
nlp_manager = NLPModelManager()
classifier = DiseaseClassifier()

# Helper to ensure NLP models are ready
async def _ensure_models():
    if not nlp_manager._loaded:
        await nlp_manager.load_models()
    if not classifier.model:
        await classifier.load_model()

async def language_detect_node(state: TriageState) -> dict:
    """Detect language from the first message."""
    await _ensure_models()
    messages = state.get("messages", [])
    if not messages:
        return {"language": "hi"}
    
    last_message = messages[-1].get("content", "")
    lang = nlp_manager.detect_language(last_message)
    return {"language": lang}


from pydantic import BaseModel, Field

class RoutingDecision(BaseModel):
    next_agent: str = Field(description="The next agent to route to: 'symptom_collector', 'classifier', or 'response_composer'")

async def brain_supervisor_node(state: TriageState) -> dict:
    """The central LLM Brain Agent evaluating patient state to coordinate specialized worker agents."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("No OPENAI_API_KEY found! Brain falling back to static pipeline.")
        return {"next_agent": "symptom_collector"}
        
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(RoutingDecision)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Clinical Brain Supervisor overseeing a rigorous hospital diagnostic agent flow.\n"
            "Assess the patient's state against the required medical thresholds:\n"
            "- Current Extracted Symptoms: {symptoms}\n"
            "- Duration of Sickness Known?: {duration}\n"
            "ROUTING DIRECTIVES:\n"
            "1. You MUST route to 'symptom_collector' IF:\n"
            "   - The duration of the sickness is NOT known.\n"
            "   - OR there are fewer than 3 symptoms, AND the patient has NOT explicitly stated 'these are the only symptoms' or 'nothing else'.\n"
            "2. Route to 'classifier' (to finalize the official diagnosis) ONLY IF:\n"
            "   - You have at least 2-3 symptoms AND know the duration of sickness.\n"
            "   - OR the patient explicitly stated they have no other symptoms to share.\n"
            "3. Route to 'response_composer' ONLY if the user is making non-medical small talk (e.g. 'hello')."),
            ("human", "Latest User Message Translation: {last_message}")
        ])
        
        chain = prompt | structured_llm
        
        messages = state.get("messages", [])
        last_message = messages[-1]["content"] if messages else ""
        symptoms = state.get("symptoms", [])
        symptoms_str = ", ".join(symptoms) if symptoms else "None"
        entities = state.get("medical_entities", {})
        duration_val = entities.get("duration", "Missing/Unknown")
        
        decision = await chain.ainvoke({
            "symptoms": symptoms_str,
            "duration": duration_val,
            "last_message": last_message,
        })
        
        next_agent = decision.next_agent
        if next_agent not in ['symptom_collector', 'classifier', 'response_composer']:
            next_agent = "symptom_collector"
            
        logger.info(f"🧠 Brain Router Delegating payload to -> [{next_agent.upper()}] WORKER AGENT.")
        return {"next_agent": next_agent}
        
    except Exception as e:
        logger.error(f"Brain API Error: {e}")
        return {"next_agent": "symptom_collector"}

async def symptom_collector_node(state: TriageState) -> dict:
    """Extract symptoms and determine if we need to ask more questions."""
    await _ensure_models()
    messages = state.get("messages", [])
    lang = state.get("language", "hi")
    
    if not messages:
        return {"symptoms": [], "question_count": 0}
        
    last_message = messages[-1].get("content", "")
    
    # Translate to English for NER
    english_text = await nlp_manager.translate_to_english(last_message, lang)
    
    # Extract entities
    entities = nlp_manager.extract_entities(english_text)
    new_symptoms = entities.get("symptoms", [])
    
    return {
        "symptoms": new_symptoms, 
        "medical_entities": entities,
        "question_count": state.get("question_count", 0) + 1
    }

async def classifier_node(state: TriageState) -> dict:
    """Predict urgency and conditions based on extracted symptoms."""
    await _ensure_models()
    symptoms = state.get("symptoms", [])
    entities = state.get("medical_entities", {})
    
    # Dummy logic for now, using the classifier stub
    features = {"symptoms": symptoms, "severity": entities.get("severity", "moderate")}
    prediction = await classifier.predict(features)
    
    # For now, map prediction dummy data
    urgency = "routine"
    if "fever" in symptoms and "severe" in entities.get("severity", ""):
        urgency = "urgent"
    elif "chest_pain" in symptoms or "breathlessness" in symptoms:
        urgency = "emergency"
        
    conditions = [{"name": prediction.get("disease_code", "viral"), "confidence": prediction.get("confidence", 0.8)}]
    
    return {"urgency": urgency, "conditions": conditions, "session_completed": True}

async def rag_advisor_node(state: TriageState) -> dict:
    """Retrieve advice from FAISS based on conditions and symptoms."""
    urgency = state.get("urgency", "routine")
    
    if urgency == "emergency":
        advice = "Please seek immediate medical attention. Do not wait."
    elif urgency == "urgent":
        advice = "You should visit the nearest clinic soon."
    else:
        advice = "Please rest and drink plenty of fluids. If symptoms worsen, consult a doctor."
        
    return {"advice": advice}

async def response_composer_node(state: TriageState) -> dict:
    """Compose the final linguistic response context using LLM Generation or static fallback."""
    await _ensure_models()
    import os
    lang = state.get("language", "hi")
    symptoms = state.get("symptoms", [])
    entities = state.get("medical_entities", {})
    advice = state.get("advice", "")
    conditions = state.get("conditions", [])
    messages = state.get("messages", [])
    next_agent = state.get("next_agent", "symptom_collector")
    
    # Base fallback logic if OpenAI is disabled
    if not os.getenv("OPENAI_API_KEY"):
        if len(symptoms) < 3:
            eng_response = "I am Swastha Sevak. To help me properly evaluate your condition, could you please describe exactly what you are feeling?"
        else:
            condition_name = conditions[0]["name"].replace("_", " ") if conditions else "a general condition"
            eng_response = f"Thank you for sharing. Based on my evaluation of your symptoms, your presentation indicates a potential risk of {condition_name}. {advice}"
    else:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-4:]])
        
        system_instructions = (
            "You are Dr. Swastha Sevak, a methodical and highly professional hospital physician. "
            "You MUST converse natively in English. Keep responses under 3 sentences.\n"
        )
        
        if next_agent == "response_composer":
            system_instructions += (
                "The target user is greeting you or making general non-medical requests. "
                "You MUST greet them back warmly, introduce yourself as 'Swastha Sevak', and ask them politely "
                "'How can I assist you with your health today?' or 'Please tell me about your symptoms'."
            )
        elif next_agent == "classifier":
            condition_name = conditions[0]['name'] if conditions else "general condition"
            system_instructions += (
                f"You have finished the patient intake. Summarize their recorded symptoms ({symptoms}) and duration ({entities.get('duration', 'unknown')}). "
                f"Declare their potential risk of {condition_name}. "
                f"You MUST include this specific medical instruction: '{advice}'."
            )
        else:
            # We are extracting variables in the loop!
            system_instructions += (
                f"Act like a fully-fledged doctor rigorously evaluating the patient's condition (Chain of Thought). "
                f"Acknowledge the symptoms recorded so far: {symptoms}. "
                f"CRITICAL MEDICAL DIRECTIVES:\n"
                f"1. If you don't know exactly 'HOW LONG' they have been sick, YOU MUST ask them how many days/weeks they have felt this way.\n"
                f"2. If you know the duration BUT have fewer than 3 symptoms, rigorously ask them if they have specific differential symptoms (e.g. fever, headaches, body pain) to gather more data.\n"
                f"3. ABSOLUTELY NEVER guess, assume, or name any potential diseases/illnesses to the patient at this stage. Stick strictly to asking about physical symptoms.\n"
                f"Ask only ONE clear follow-up question."
            )
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            ("human", "Patient History Context:\n{history}\n\nDraft your clinical verbal reply to the patient now:")
        ])
        
        try:
            chain = prompt | llm
            response = await chain.ainvoke({"history": history})
            eng_response = response.content.strip()
        except Exception as e:
            logger.error(f"Response Composer LLM Error: {e}")
            eng_response = "Could you please elaborate on your symptoms?"

    # Translate the dynamic English output securely back into the patient's native dialect!
    final_reply = await nlp_manager.translate_from_english(eng_response, lang)
    
    return {
        "messages": [{"role": "assistant", "content": final_reply}]
    }

async def escalation_node(state: TriageState) -> dict:
    """Alert doctor if emergency."""
    urgency = state.get("urgency", "routine")
    escalated = False
    if urgency in ["emergency", "urgent"]:
        # Mock escalation
        escalated = True
        
    return {"escalated": escalated, "nearest_phc": {"name": "Central Hospital", "distance": "5km", "phone": "108"}}
