"""Quick integration test for the triage agent graph with mocks."""

from __future__ import annotations

import os
import sys

# Enable mocks for testing
os.environ["MOCK_NLP"] = "true"
os.environ["MOCK_CLASSIFIER"] = "true"


def test_graph_imports() -> None:
    """Verify all graph components can be imported."""
    from app.agent.states import TriageState
    from app.agent.graph import triage_graph
    from app.agent.nodes.language_detect import language_detect_node
    from app.agent.nodes.symptom_collector import symptom_collector_node
    from app.agent.nodes.classifier import classifier_node
    from app.agent.nodes.rag_advisor import rag_advisor_node
    from app.agent.nodes.response_composer import response_composer_node
    from app.agent.nodes.escalation import escalation_node

    print("✅ All imports successful")
    assert triage_graph is not None
    print("✅ Graph compiled successfully")


def test_mock_nlp() -> None:
    """Test mock NLP functions."""
    from app.nlp.models import (
        detect_language,
        process_message,
        translate_to_en,
        translate_to_patient_lang,
    )

    # Test language detection
    lang = detect_language("I have fever")
    assert lang == "hi", f"Expected 'hi', got '{lang}'"
    print(f"✅ detect_language() → '{lang}'")

    # Test symptom extraction
    symptoms, entities = process_message("I have fever and cough")
    assert symptoms == ["fever", "cough"], f"Expected ['fever', 'cough'], got {symptoms}"
    print(f"✅ process_message() → symptoms={symptoms}")

    # Test translation to English
    text_en = translate_to_en("I have fever", "hi")
    assert isinstance(text_en, str)
    print(f"✅ translate_to_en() → '{text_en}'")

    # Test translation to patient language
    text_hi = translate_to_patient_lang("I have fever", "hi")
    assert "[hi]" in text_hi, f"Expected '[hi]' in '{text_hi}'"
    print(f"✅ translate_to_patient_lang() → contains language code")


def test_mock_classifier() -> None:
    """Test mock classifier function."""
    from app.ml.classifier import predict_triage

    urgency, conditions = predict_triage(symptoms=["fever", "cough"])
    assert urgency == "routine", f"Expected 'routine', got '{urgency}'"
    assert len(conditions) == 3, f"Expected 3 conditions, got {len(conditions)}"
    assert conditions[0]["name"] == "Common Cold"
    print(f"✅ predict_triage() → urgency='{urgency}', conditions={len(conditions)}")


def test_graph_execution() -> None:
    """Test full graph execution with mock data."""
    from app.agent.graph import triage_graph
    from app.agent.states import TriageState

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

    print("\n📊 Running graph with mock data...")
    output = triage_graph.invoke(state)

    # Verify output
    assert output["language"] in ["hi", "en"]
    print(f"  ✓ Language: {output['language']}")

    assert len(output["symptoms"]) > 0
    print(f"  ✓ Symptoms detected: {output['symptoms']}")

    assert output["urgency"] in ["self_care", "routine", "urgent", "emergency"]
    print(f"  ✓ Urgency: {output['urgency']}")

    assert len(output["conditions"]) > 0
    print(f"  ✓ Conditions: {[c['name'] for c in output['conditions'][:2]]}")

    assert len(output["advice"]) > 0
    print(f"  ✓ Advice provided ({len(output['advice'])} chars)")

    assert len(output["messages"]) > 1
    print(f"  ✓ Messages exchanged: {len(output['messages'])}")

    final_message = output["messages"][-1]["content"]
    print(f"\n💬 Final bot response:\n{final_message[:200]}...")


if __name__ == "__main__":
    print("=" * 70)
    print("TRIAGE AGENT TEST SUITE (MOCK MODE)")
    print("=" * 70)

    try:
        print("\n[1/4] Testing imports...")
        test_graph_imports()

        print("\n[2/4] Testing mock NLP...")
        test_mock_nlp()

        print("\n[3/4] Testing mock classifier...")
        test_mock_classifier()

        print("\n[4/4] Testing full graph execution...")
        test_graph_execution()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
