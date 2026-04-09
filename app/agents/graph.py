from typing import Any, Dict
from langgraph.graph import StateGraph, END
import logging

from app.agents.state import TriageState
from app.agents.nodes import (
    language_detect_node,
    brain_supervisor_node,
    symptom_collector_node,
    classifier_node,
    rag_advisor_node,
    response_composer_node,
    escalation_node,
    nlp_manager,
    classifier
)

logger = logging.getLogger(__name__)

def route_from_brain(state: TriageState) -> str:
    """Conditional edge checking the brain's internal routing token variable map."""
    return state.get("next_agent", "symptom_collector")

def check_escalation(state: TriageState) -> str:
    """Conditional edge to see if we should escalate."""
    urgency = state.get("urgency", "routine")
    if urgency in ["emergency", "urgent"]:
        return "escalation"
    return END

def build_graph() -> StateGraph:
    """Build and compile the LangGraph Hub-and-Spoke Hub StateMachine."""
    workflow = StateGraph(TriageState)
    
    workflow.add_node("language_detect", language_detect_node)
    workflow.add_node("brain", brain_supervisor_node)
    workflow.add_node("symptom_collector", symptom_collector_node)
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("rag_advisor", rag_advisor_node)
    workflow.add_node("response_composer", response_composer_node)
    workflow.add_node("escalation", escalation_node)
    
    # ── Entry ──
    workflow.set_entry_point("language_detect")
    
    # ── Lang to Brain ──
    workflow.add_edge("language_detect", "brain")
    
    # ── Brain Dispatch Router ──
    workflow.add_conditional_edges(
        "brain",
        route_from_brain,
        {
            "symptom_collector": "symptom_collector",
            "classifier": "classifier",
            "response_composer": "response_composer"
        }
    )
    
    # ── Worker Agent Feedbacks ──
    # If symptom collected, send to response composer logically to build reply context
    workflow.add_edge("symptom_collector", "response_composer")
    
    # Classifier to RAG
    workflow.add_edge("classifier", "rag_advisor")
    
    # RAG to composer
    workflow.add_edge("rag_advisor", "response_composer")
    
    # ── Escalate logic ──
    workflow.add_conditional_edges(
        "response_composer",
        check_escalation,
        {
            "escalation": "escalation",
            END: END
        }
    )
    
    workflow.add_edge("escalation", END)
    
    return workflow.compile()


class TriageAgent:
    """Agent orchestrator that matches the API of the previous stub."""
    
    def __init__(self) -> None:
        self.graph = build_graph()
        self.checkpoint: dict[str, Any] = {}
        
    async def load_model(self) -> None:
        await nlp_manager.load_models()
        await classifier.load_model()
        logger.info("TriageAgent loaded models")

    async def shutdown(self) -> None:
        await nlp_manager.shutdown()
        await classifier.shutdown()
        logger.info("TriageAgent initialized shutdown")

    async def initialize(self) -> None:
        self.checkpoint = {}

    async def run(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        """Run the graph given the patient input."""
        logger.info(f"Running agent with input: {patient_data}")
        
        # Prepare initial state
        initial_state = {
            "patient_phone": patient_data.get("phone", ""),
            "messages": [{"role": "user", "content": patient_data.get("message", "")}],
        }
        
        # If we have an existing session checkpoint, we could merge state here
        if "state" in self.checkpoint:
            initial_state = {**self.checkpoint["state"], **initial_state}
            
        final_state = await self.graph.ainvoke(initial_state)
        
        # Save checkpoint
        self.checkpoint["state"] = final_state
        
        return {
            "triage_level": final_state.get("urgency", "unknown"),
            "next_step": "handoff" if final_state.get("escalated") else "awaiting_data",
            "reply": final_state.get("messages", [])[-1]["content"] if final_state.get("messages") else "",
            "state": final_state
        }
