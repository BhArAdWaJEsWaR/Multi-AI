import unittest

import numpy as np

from core.orchestrator import (
    Orchestrator,
    _looks_like_coding_query,
    _looks_like_math_query,
    _looks_like_writing_query,
)


class FakeMemory:
    def __init__(self):
        self.entries = [
            ("recursive", {"type": "preference_update", "pref_key": "recursive_approach", "domain": "coding"}),
            ("stale recursive", {"type": "preference_update", "pref_key": "recursive_approach", "domain": "coding"}),
            ("global", {"type": "preference_update", "pref_key": "format", "domain": "all"}),
            ("writing", {"type": "preference_update", "pref_key": "tone", "domain": "writing"}),
        ]
        self.removed = []

    def get_by_metadata(self, where):
        return list(self.entries)

    def remove_by_metadata(self, where):
        self.removed.append(where)
        return 1


class PreferenceResetTests(unittest.TestCase):
    def make_orchestrator(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.use_memory = True
        orchestrator.memory = FakeMemory()
        orchestrator._conversation_history = [
            {"role": "user", "content": "USER PREFERENCES FOR CODING (MANDATORY): Use recursion."},
            {"role": "assistant", "content": "Here is recursive code."},
            {"role": "user", "content": "USER PREFERENCES FOR WRITING (MANDATORY): Be concise."},
            {"role": "assistant", "content": "I will be concise."},
            {"role": "user", "content": "USER PREFERENCES FOR ALL (MANDATORY): Use plain language."},
            {"role": "assistant", "content": "I will use plain language."},
            {"role": "user", "content": "Unrelated follow-up"},
            {"role": "assistant", "content": "Unrelated answer"},
        ]
        return orchestrator

    def test_forget_domain_syntax_is_detected(self):
        orchestrator = self.make_orchestrator()

        update = orchestrator._detect_preference_update(
            "forget all my preferences in coding"
        )

        self.assertEqual(update, {"type": "RESET", "domain": "coding"})

    def test_explicit_preference_setting_remains_unchanged(self):
        orchestrator = self.make_orchestrator()

        update = orchestrator._detect_preference_update(
            "I prefer recursion over iteration when writing code"
        )

        self.assertEqual(update["type"], "SET")
        self.assertEqual(update["domain"], "coding")
        self.assertIn("recursion", update["instruction"].lower())

    def test_domain_reset_removes_stale_records_and_session_context(self):
        orchestrator = self.make_orchestrator()

        orchestrator._active_preferences = {
            "recursive_approach": {"domain": "coding", "instruction": "Use recursion."},
            "tone": {"domain": "writing", "instruction": "Be concise."},
        }
        orchestrator._active_preferences = {
            key: value
            for key, value in orchestrator._active_preferences.items()
            if value["domain"] != "coding"
        }
        removed = orchestrator._remove_persisted_preferences("coding")
        orchestrator._clear_preference_context("coding")

        self.assertEqual(removed, 2)
        self.assertEqual(list(orchestrator._active_preferences), ["tone"])
        self.assertEqual(
            [item["content"] for item in orchestrator._conversation_history],
            [
                "USER PREFERENCES FOR WRITING (MANDATORY): Be concise.",
                "I will be concise.",
                "Unrelated follow-up",
                "Unrelated answer",
            ],
        )
        self.assertEqual(
            {item["domain"] for item in orchestrator.memory.removed},
            {"coding", "all"},
        )


class RoutingRegressionTests(unittest.TestCase):
    def make_orchestrator(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._last_embedding = np.array([1.0, 0.0])
        orchestrator._last_task_type = "coding"
        orchestrator._same_topic_streak = 0
        return orchestrator

    def test_explicit_email_request_wins_over_stale_topic(self):
        orchestrator = self.make_orchestrator()

        task_type, method, dataset = orchestrator.classify(
            "write an email for my friend regarding invitation to my engagement",
            np.array([1.0, 0.0]),
            None,
        )

        self.assertEqual((task_type, method, dataset), ("writing", "keyword_writing", None))

    def test_write_code_still_routes_to_coding(self):
        orchestrator = self.make_orchestrator()

        self.assertTrue(_looks_like_coding_query("write code for bubble sort"))
        self.assertFalse(_looks_like_writing_query("write code for bubble sort"))
        self.assertTrue(_looks_like_writing_query("write an email about my class schedule"))
        self.assertTrue(_looks_like_coding_query("write an if statement"))
        self.assertFalse(_looks_like_writing_query("write a function that sends an email"))

    def test_derivative_expression_routes_to_math(self):
        orchestrator = self.make_orchestrator()

        self.assertTrue(_looks_like_math_query("what is the derivative of 3x"))
        task_type, method, dataset = orchestrator.classify(
            "what is the derivative of 3x",
            np.array([1.0, 0.0]),
            None,
        )

        self.assertEqual((task_type, method, dataset), ("math", "keyword_math", None))


if __name__ == "__main__":
    unittest.main()
