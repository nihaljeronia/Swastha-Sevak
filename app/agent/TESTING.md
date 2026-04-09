"""Testing guide for the triage agent nodes and graph.

This document explains how to test the triage agent with mock implementations
before Devang's NLP and Navya's ML classifier are fully integrated.
"""

# ============================================================================
# ENVIRONMENT VARIABLES FOR MOCK MODE
# ============================================================================

# Set these environment variables to enable mock implementations:
#
#   MOCK_NLP=true       - Use mock NLP functions (language detection, NER, translation)
#   MOCK_CLASSIFIER=true - Use mock ML classifier (triage prediction)
#
# Example:
#   $ export MOCK_NLP=true
#   $ export MOCK_CLASSIFIER=true
#   $ python -m uvicorn app.main:app --reload
#
# Or in PowerShell:
#   $env:MOCK_NLP='true'
#   $env:MOCK_CLASSIFIER='true'

# ============================================================================
# MOCK BEHAVIOR
# ============================================================================

# When MOCK_NLP=true:
#   - detect_language() → always returns "hi" (Hindi)
#   - process_message() → always returns ["fever", "cough"] symptoms
#   - translate_to_en() → returns text as-is (assumes input is descriptive)
#   - translate_to_patient_lang() → appends "[hi]" to show translation happened

# When MOCK_CLASSIFIER=true:
#   - predict_triage() → always returns:
#     * urgency = "routine"
#     * conditions = ["Common Cold", "URI", "Influenza"]

# ============================================================================
# TESTING THE GRAPH
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import os
    from app.agent.graph import triage_graph
    from app.agent.states import TriageState

    # Enable mock mode
    os.environ["MOCK_NLP"] = "true"
    os.environ["MOCK_CLASSIFIER"] = "true"

    async def test_triage():
        """Run a simple triage session end-to-end."""
        state = TriageState(
            patient_phone="919876543210",
            language="en",
            messages=[
                {
                    "role": "user",
                    "content": "I have fever and cough for 3 days",
                    "language": "en",
                }
            ],
            symptoms=[],
            medical_entities={},
            question_count=0,
            urgency="",
            conditions=[],
            advice="",
            nearest_phc={},
            follow_up_scheduled=False,
        )

        # Invoke the graph
        output = triage_graph.invoke(state)

        # Print results
        print("\n" + "=" * 70)
        print("TRIAGE RESULT")
        print("=" * 70)
        print(f"Patient phone: {output['patient_phone']}")
        print(f"Language: {output['language']}")
        print(f"Symptoms: {output['symptoms']}")
        print(f"Medical entities: {output['medical_entities']}")
        print(f"Urgency: {output['urgency']}")
        print(f"Conditions: {output['conditions']}")
        print(f"Advice: {output['advice']}")
        print(f"Nearest PHC: {output['nearest_phc']}")
        print("\n" + "=" * 70)
        print("CONVERSATION")
        print("=" * 70)
        for msg in output["messages"]:
            role = "👤 Patient" if msg["role"] == "user" else "🤖 Bot"
            print(f"\n{role}:")
            print(f"  {msg['content']}")

    asyncio.run(test_triage())

# ============================================================================
# TESTING INDIVIDUAL NODES
# ============================================================================

# Example: Test the classifier node in isolation

if __name__ == "__main__":
    from app.agent.nodes.classifier import classifier_node
    from app.agent.states import TriageState

    state = TriageState(
        patient_phone="919876543210",
        language="hi",
        messages=[],
        symptoms=["fever", "cough", "sore_throat"],
        medical_entities={"body_parts": ["throat"], "severity": ["moderate"]},
        question_count=2,
        urgency="",
        conditions=[],
        advice="",
        nearest_phc={},
        follow_up_scheduled=False,
    )

    result = classifier_node(state)
    print(f"Classifier result: {result}")

# ============================================================================
# TESTING WITHOUT MOCKS
# ============================================================================

# When real implementations are available, just unset the env vars:
#
#   $ unset MOCK_NLP
#   $ unset MOCK_CLASSIFIER
#
# Or set them to false/off:
#
#   $ export MOCK_NLP=false
#   $ export MOCK_CLASSIFIER=false
#
# The functions will automatically fall back to real implementations
# (or return warnings if not yet implemented).

# ============================================================================
# INTEGRATION WITH WEBHOOK
# ============================================================================

# The webhook (app/webhook/routes.py) will invoke the graph like this:
#
#   output_state = triage_graph.invoke(input_state)
#   response_message = output_state["messages"][-1]["content"]
#   # Send response_message back to WhatsApp via Meta API

print(__doc__)
