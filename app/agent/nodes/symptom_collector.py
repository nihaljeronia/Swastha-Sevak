"""Symptom collection via follow-up questions node."""

from __future__ import annotations

import logging

from app.agent.states import TriageState
from app.nlp.models import process_message, translate_to_en, translate_to_patient_lang

logger = logging.getLogger(__name__)


def generate_followup_question(symptoms: list[str], question_num: int) -> str:
    """Generate contextual follow-up question based on current symptoms.

    Args:
        symptoms: Current list of extracted symptoms.
        question_num: Which follow-up question number (1-5).

    Returns:
        Follow-up question in English.
    """
    if not symptoms:
        questions = [
            "What symptoms are you experiencing?",
            "When did these symptoms start?",
            "How severe would you rate the pain/discomfort?",
            "Are you experiencing any other symptoms?",
            "Have you taken any medications?",
        ]
    else:
        symptom_str = ", ".join(symptoms[:min(2, len(symptoms))])
        questions = [
            f"You mentioned {symptom_str}. How long have you had this?",
            "Is the pain/discomfort constant or does it come and go?",
            "Have you had a fever or chills?",
            "Any nausea, vomiting, or changes in appetite?",
            "Are you taking any medications currently?",
        ]

    question_idx = min(question_num - 1, len(questions) - 1)
    return questions[question_idx]


def symptom_collector_node(state: TriageState) -> dict:
    """Collect symptoms via follow-up questions or proceed to classifier.

    Implements conditional logic:
    - If len(symptoms) < 3 AND question_count < 5: ask follow-up question
    - If len(symptoms) >= 3 OR question_count >= 5 OR patient says "bas": proceed
    - Patient should respond with "bas" / "that's all" / "no more" to stop early

    Args:
        state: Current triage state.

    Returns:
        Partial state update dict. If asking a question, adds to messages and
        increments question_count. If proceeding to classifier, returns empty dict
        (which signals via routing logic).
    """
    logger.info(
        f"[symptom_collector] Q#{state['question_count']}, "
        f"{len(state['symptoms'])} symptoms"
    )

    symptoms_count = len(state["symptoms"])
    question_count = state["question_count"]

    # Check if patient explicitly said they're done
    if state["messages"]:
        last_message = state["messages"][-1]
        last_content = last_message.get("content", "").lower() if last_message else ""
        patient_done_phrases = [
            "bas",
            "that's all",
            "no more",
            "thats all",
            "done",
            "nothing else",
        ]

        if any(phrase in last_content for phrase in patient_done_phrases):
            logger.info("[symptom_collector] Patient explicitly finished collection")
            return {}

    # If enough symptoms or max questions, proceed to classifier (signal via empty dict)
    if symptoms_count >= 3:
        logger.info("[symptom_collector] Enough symptoms, proceeding to classifier")
        return {}

    if question_count >= 5:
        logger.info("[symptom_collector] Max questions reached, proceeding to classifier")
        return {}

    # Generate and ask follow-up question
    logger.info("[symptom_collector] Generating follow-up question")
    question_en = generate_followup_question(state["symptoms"], question_count + 1)
    question_lang = translate_to_patient_lang(question_en, state["language"])

    # Add assistant message for the question
    messages = state["messages"].copy() if state["messages"] else []
    messages.append(
        {
            "role": "assistant",
            "content": question_lang,
            "language": state["language"],
        }
    )

    return {
        "messages": messages,
        "question_count": question_count + 1,
    }


def should_continue_collecting_symptoms(state: TriageState) -> str:
    """Conditional edge: determine if symptom collection should continue.

    Routes to 'symptom_collector' if more questions needed, 'classifier' otherwise.

    Args:
        state: Current triage state.

    Returns:
        Node name: 'symptom_collector' or 'classifier'.
    """
    symptoms_count = len(state["symptoms"])
    question_count = state["question_count"]

    logger.info(
        f"[should_continue_collecting_symptoms] {len(state['symptoms'])} symptoms, "
        f"{question_count} questions"
    )

    # Check for patient "done" signals
    if state["messages"]:
        last_message = state["messages"][-1]
        last_content = last_message.get("content", "").lower() if last_message else ""
        patient_done_phrases = [
            "bas",
            "that's all",
            "no more",
            "thats all",
            "done",
            "nothing else",
        ]

        if any(phrase in last_content for phrase in patient_done_phrases):
            logger.info("[should_continue_collecting_symptoms] Patient done → classifier")
            return "classifier"

    # If enough symptoms, proceed to classifier
    if symptoms_count >= 3:
        logger.info("[should_continue_collecting_symptoms] 3+ symptoms → classifier")
        return "classifier"

    # If max questions reached, proceed to classifier
    if question_count >= 5:
        logger.info("[should_continue_collecting_symptoms] Max Q reached → classifier")
        return "classifier"

    # Continue collecting symptoms
    logger.info("[should_continue_collecting_symptoms] Continue → symptom_collector")
    return "symptom_collector"
