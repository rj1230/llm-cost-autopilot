"""
Unit tests for the Phase 3 verification logic. Everything here is mocked or
pure-function - no API keys or network required, so this can (and should)
run in CI. Run with:

    python -m unittest tests.test_verification -v
"""

import unittest
from unittest.mock import patch

from src.models.response import Response
from src.verification.scoring import jaccard_similarity, label_match, parse_judge_score
from src.verification.task_type import infer_task_type
from src.verification.verifier import verify_response


def _fake_response(text: str, cost: float = 0.001, model_name: str = "m") -> Response:
    return Response(
        output_text=text,
        input_tokens=10,
        output_tokens=10,
        latency_s=0.1,
        cost_usd=cost,
        model_name=model_name,
        provider="test",
    )


class TestScoring(unittest.TestCase):
    def test_jaccard_identical(self):
        self.assertEqual(jaccard_similarity("raj418060@gmail.com", "raj418060@gmail.com"), 1.0)

    def test_jaccard_disjoint(self):
        self.assertEqual(jaccard_similarity("apple banana", "car train"), 0.0)

    def test_jaccard_partial(self):
        score = jaccard_similarity("the quick brown fox", "the slow brown fox")
        self.assertTrue(0.0 < score < 1.0)

    def test_label_match_same_case_insensitive(self):
        self.assertEqual(label_match("Positive", "positive"), 1.0)

    def test_label_match_different(self):
        self.assertEqual(label_match("positive", "negative"), 0.0)

    def test_parse_judge_score_clean_digit(self):
        self.assertEqual(parse_judge_score("5"), 1.0)
        self.assertEqual(parse_judge_score("3"), 0.6)

    def test_parse_judge_score_sentence(self):
        self.assertEqual(parse_judge_score("I'd rate this a 4 out of 5."), 4 / 5)

    def test_parse_judge_score_unparseable_defaults_soft(self):
        self.assertEqual(parse_judge_score("no clear number here"), 0.6)


class TestTaskType(unittest.TestCase):
    def test_extraction(self):
        self.assertEqual(
            infer_task_type("Extract the email address from this text: '...'"),
            "extraction",
        )

    def test_classification(self):
        self.assertEqual(
            infer_task_type("Classify the sentiment of this review as positive or negative"),
            "classification",
        )

    def test_summarization(self):
        self.assertEqual(
            infer_task_type("Summarize the following in two sentences: '...'"),
            "summarization",
        )

    def test_other(self):
        self.assertEqual(
            infer_task_type("Design a system architecture for a ride-sharing app"),
            "other",
        )


class TestVerifyResponse(unittest.TestCase):
    @patch("src.verification.verifier._reference_model_name", return_value="gpt-4o")
    @patch("src.verification.verifier.send_request")
    def test_extraction_pass_no_escalation(self, mock_send, _mock_ref_name):
        mock_send.return_value = _fake_response("raj418060@gmail.com", cost=0.002, model_name="gpt-4o")
        cheap = _fake_response("raj418060@gmail.com", cost=0.0001, model_name="local-llama")

        outcome = verify_response(
            "Extract the email address from: 'Contact raj418060@gmail.com'",
            cheap,
            "local-llama",
            "req-1",
        )

        self.assertTrue(outcome.passed)
        self.assertFalse(outcome.escalated)
        self.assertEqual(outcome.cost_delta_usd, 0.0)
        mock_send.assert_called_once()  # extraction never calls the judge prompt

    @patch("src.verification.verifier._reference_model_name", return_value="gpt-4o")
    @patch("src.verification.verifier.send_request")
    def test_classification_fail_triggers_escalation(self, mock_send, _mock_ref_name):
        mock_send.return_value = _fake_response("negative", cost=0.003, model_name="gpt-4o")
        cheap = _fake_response("positive", cost=0.0002, model_name="local-llama")

        outcome = verify_response(
            "Classify the sentiment: 'meh, it was fine I guess'",
            cheap,
            "local-llama",
            "req-2",
        )

        self.assertFalse(outcome.passed)
        self.assertTrue(outcome.escalated)
        self.assertAlmostEqual(outcome.cost_delta_usd, 0.003 - 0.0002)
        self.assertEqual(outcome.final_response.output_text, "negative")

    @patch("src.verification.verifier._reference_model_name", return_value="gpt-4o")
    @patch("src.verification.verifier.send_request")
    def test_summarization_calls_judge_and_passes(self, mock_send, _mock_ref_name):
        # first call -> reference model's own answer; second call -> judge score
        mock_send.side_effect = [
            _fake_response("A two sentence summary.", cost=0.004, model_name="gpt-4o"),
            _fake_response("5", cost=0.0005, model_name="gpt-4o"),
        ]
        cheap = _fake_response("A similar two sentence summary.", cost=0.0003, model_name="gpt-4o-mini")

        outcome = verify_response(
            "Summarize the following in two sentences: '...'", cheap, "gpt-4o-mini", "req-3"
        )

        self.assertEqual(mock_send.call_count, 2)
        self.assertTrue(outcome.passed)
        self.assertFalse(outcome.escalated)


if __name__ == "__main__":
    unittest.main()
