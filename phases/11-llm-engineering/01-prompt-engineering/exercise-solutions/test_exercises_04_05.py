import unittest
from unittest.mock import patch

import exercise_04 as optimizer
import exercise_05 as prompt_diff


class OptimizerTests(unittest.TestCase):
    def test_three_rounds_with_five_samples_each(self):
        calls = []

        def request(prompt):
            calls.append(prompt)
            return "{}" if "Return only valid JSON" in prompt else "plain text"

        result = optimizer.optimize_prompt("Explain limits", {"expected_format": "json"}, request)
        self.assertEqual(len(calls), 20)
        self.assertEqual(len(result["history"]), 4)
        self.assertTrue(all(len(stage["samples"]) == 5 for stage in result["history"]))
        self.assertEqual(result["score_change"], 1.0)
        self.assertEqual(result["history"][1]["targeted_criterion"], "format_compliant")
        self.assertEqual(result["history"][0]["prompt"], "Explain limits")

    def test_no_improvement_is_reported(self):
        result = optimizer.optimize_prompt(
            "Explain limits", {"expected_format": "json"}, lambda prompt: "not JSON"
        )
        self.assertEqual(result["score_change"], 0)
        self.assertEqual(result["best"]["prompt"], "Explain limits")

    def test_weakest_criterion_is_targeted(self):
        result = optimizer.optimize_prompt(
            "Explain limits", {"max_words": 10, "required_keywords": ["API"]},
            lambda prompt: "brief response",
        )
        self.assertEqual(result["history"][1]["targeted_criterion"], "keyword_coverage")
        self.assertIn("Include each of these terms explicitly: API.", result["history"][1]["prompt"])

    def test_rejects_invalid_criteria_and_empty_outputs(self):
        for criteria in ({}, {"max_words": 0}, {"expected_format": "unknown"}):
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                optimizer.evaluate_prompt("Task", criteria, lambda prompt: "text")
        with self.assertRaises(RuntimeError):
            optimizer.evaluate_prompt("Task", {"max_words": 10}, lambda prompt: " ")

    def test_temperature_and_truncation(self):
        response = {"response": "text", "finish_reason": "stop"}
        with patch.object(optimizer, "ollama_request", return_value=response) as request:
            self.assertEqual(optimizer.request_output("Task"), "text")
            self.assertEqual(request.call_args.args[0]["temperature"], 0.7)
            self.assertEqual(request.call_args.args[2], [{"role": "user", "content": "Task"}])
        with patch.object(optimizer, "ollama_request", return_value={**response, "finish_reason": "length"}):
            with self.assertRaises(RuntimeError):
                optimizer.request_output("Task")


class PromptDiffTests(unittest.TestCase):
    def setUp(self):
        self.before = {"task": "Explain limits"}
        self.after = {**self.before, "format": "JSON"}

    def test_added_constraints_and_format(self):
        result = prompt_diff.diff_prompts(
            self.before, {**self.after, "constraints": ["Use at most 10 words"]}
        )
        self.assertEqual(result["prediction"], "improve")
        self.assertEqual({change["field"] for change in result["changes"]}, {"format", "constraints"})
        self.assertIn("+Output format: JSON", result["text_diff"])

    def test_removed_examples(self):
        result = prompt_diff.diff_prompts({**self.before, "examples": ["An example"]}, self.before)
        self.assertEqual(result["prediction"], "degrade")
        self.assertEqual(result["changes"][0]["removed"], ["An example"])

    def test_role_and_format_replacements_are_uncertain(self):
        result = prompt_diff.diff_prompts(
            {**self.before, "role": "Teacher", "format": "JSON"},
            {**self.before, "role": "Editor", "format": "Bullets"},
        )
        self.assertEqual(result["prediction"], "uncertain")

    def test_unchanged_prompt(self):
        result = prompt_diff.diff_prompts(self.before, self.before)
        self.assertEqual(result["prediction"], "unchanged")
        self.assertEqual(result["text_diff"], "")

    def test_paired_outputs_confirm_prediction(self):
        calls = []

        def request(prompt):
            calls.append(prompt)
            return "{}" if "Output format: JSON" in prompt else "plain text"

        result = prompt_diff.test_prediction(
            self.before, self.after, {"expected_format": "json"}, request
        )
        self.assertEqual(len(calls), 10)
        self.assertEqual(calls[0], prompt_diff.render_prompt(self.before))
        self.assertEqual(calls[1], prompt_diff.render_prompt(self.after))
        self.assertTrue(result["prediction_correct"])
        self.assertEqual(result["score_change"], 1)

    def test_prediction_can_be_wrong(self):
        result = prompt_diff.test_prediction(
            self.before, self.after, {"expected_format": "json"},
            lambda prompt: "not JSON" if "Output format: JSON" in prompt else "{}",
        )
        self.assertEqual(result["observed"], "degrade")
        self.assertFalse(result["prediction_correct"])

    def test_rejects_task_change_and_invalid_inputs(self):
        with self.assertRaises(ValueError):
            prompt_diff.test_prediction(
                self.before, {"task": "Different task"}, {"max_words": 10}, lambda prompt: "text"
            )
        with self.assertRaises(ValueError):
            prompt_diff.render_prompt({**self.before, "examples": "not a list"})
        with self.assertRaises(ValueError):
            prompt_diff.test_prediction(
                self.before, self.after, {"max_words": 10}, lambda prompt: "text", runs=0
            )


if __name__ == "__main__":
    unittest.main()