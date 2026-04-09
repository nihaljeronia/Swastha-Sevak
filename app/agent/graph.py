"""LangGraph state machine for medical triage workflow.

This module defines the complete triage agent graph with six nodes
and conditional routing to handle multi-turn symptom collection,
disease classification, medical advice retrieval, and escalation.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from app.agent.states import TriageState
from app.agent.nodes.language_detect import language_detect_node
from app.agent.nodes.symptom_collector import (
    symptom_collector_node,
    should_continue_collecting_symptoms,
)
from app.agent.nodes.classifier import classifier_node
from app.agent.nodes.rag_advisor import rag_advisor_node
from app.agent.nodes.response_composer import response_composer_node
from app.agent.nodes.escalation import escalation_node, should_escalate

logger = logging.getLogger(__name__)


# ============================================================================
# BUILD THE GRAPH
# ============================================================================


def build_triage_graph() -> StateGraph:
    """Construct the complete triage workflow graph.

    Returns:
        Compiled LangGraph StateGraph ready for execution.
    """
    graph = StateGraph(TriageState)

    # Add all nodes
    graph.add_node("language_detect", language_detect_node)
    graph.add_node("symptom_collector", symptom_collector_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("rag_advisor", rag_advisor_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("response_composer", response_composer_node)

    # Edges
    graph.add_edge(START, "language_detect")
    graph.add_edge("language_detect", "symptom_collector")

    # Conditional edge from symptom_collector
    graph.add_conditional_edges(
        "symptom_collector",
        should_continue_collecting_symptoms,
        {
            "symptom_collector": "symptom_collector",
            "classifier": "classifier",
        },
    )

    # Linear edges for classification pipeline
    graph.add_edge("classifier", "rag_advisor")
    graph.add_edge("rag_advisor", "escalation")

    # Conditional edge from escalation
    graph.add_conditional_edges(
        "escalation",
        should_escalate,
        {
            "response_composer": "response_composer",
        },
    )

    # Final edge
    graph.add_edge("response_composer", END)

    return graph.compile()


# Export the compiled graph
triage_graph = build_triage_graph()
