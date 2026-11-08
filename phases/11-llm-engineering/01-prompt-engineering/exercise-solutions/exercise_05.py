# Problem Statement : Create a "prompt diff" tool. 
# Given two versions of a prompt, identify what changed 
# (added constraints, removed examples, changed role, 
# modified format) and predict whether the change will 
# improve or degrade output quality. Test your predictions 
# against actual outputs.

import difflib
import json

from exercise_04 import evaluate_prompt


def validate_version(version):
	allowed = {"role", "task", "constraints", "examples", "format"}
	if set(version) - allowed:
		raise ValueError("Unknown prompt fields")
	if not isinstance(version.get("task"), str) or not version["task"].strip():
		raise ValueError("Each prompt needs a nonempty task")
	for field in ("role", "format"):
		if not isinstance(version.get(field, ""), str):
			raise ValueError(f"{field} must be a string")
	for field in ("constraints", "examples"):
		values = version.get(field, [])
		if not isinstance(values, list) or any(
			not isinstance(value, str) or not value.strip() for value in values
		):
			raise ValueError(f"{field} must be a list of nonempty strings")


def render_prompt(version):
	validate_version(version)
	sections = []
	if version.get("role"):
		sections.append(f"Role: {version['role']}")
	sections.append(f"Task: {version['task']}")
	if version.get("constraints"):
		sections.append("Constraints:\n" + "\n".join(
			f"- {constraint}" for constraint in version["constraints"]
		))
	if version.get("examples"):
		sections.append("Examples:\n" + "\n\n".join(version["examples"]))
	if version.get("format"):
		sections.append(f"Output format: {version['format']}")
	return "\n\n".join(sections)


def diff_prompts(before, after):
	before_text = render_prompt(before)
	after_text = render_prompt(after)
	changes = []
	signals = []
	for field in ("role", "task", "format"):
		old = before.get(field, "")
		new = after.get(field, "")
		if old != new:
			changes.append({"field": field, "before": old, "after": new})
			signals.append(0 if old and new else (1 if new else -1))
	for field in ("constraints", "examples"):
		old = before.get(field, [])
		new = after.get(field, [])
		if old == new:
			continue
		added = [value for value in new if value not in old]
		removed = [value for value in old if value not in new]
		changes.append({"field": field, "added": added, "removed": removed})
		if added:
			signals.append(1)
		if removed:
			signals.append(-1)
		if not added and not removed:
			signals.append(0)

	if not changes:
		prediction = "unchanged"
		reason = "The rendered prompts are identical."
	elif any(change["field"] == "task" for change in changes):
		prediction = "uncertain"
		reason = "The task changed; quality scores may no longer be comparable."
	elif all(signal == 1 for signal in signals):
		prediction = "improve"
		reason = "Added guidance may improve adherence if it is relevant and consistent."
	elif all(signal == -1 for signal in signals):
		prediction = "degrade"
		reason = "Removed guidance may weaken adherence to the evaluation criteria."
	else:
		prediction = "uncertain"
		reason = "Replacements, reordering, or mixed additions/removals need measured evaluation."
	return {
		"changes": changes,
		"prediction": prediction,
		"reason": reason,
		"text_diff": "\n".join(difflib.unified_diff(
			before_text.splitlines(), after_text.splitlines(),
			fromfile="before", tofile="after", lineterm="",
		)),
	}


def test_prediction(before, after, criteria, request=None, runs=5, tolerance=0.01):
	if tolerance < 0:
		raise ValueError("tolerance must be nonnegative")
	report = diff_prompts(before, after)
	if before["task"] != after["task"]:
		raise ValueError("Keep the same task to compare output quality")
	before_prompt = render_prompt(before)
	after_prompt = render_prompt(after)
	before_samples = []
	after_samples = []
	for _ in range(runs):
		before_samples.extend(evaluate_prompt(before_prompt, criteria, request, runs=1)["samples"])
		after_samples.extend(evaluate_prompt(after_prompt, criteria, request, runs=1)["samples"])
	if not before_samples:
		raise ValueError("runs must be positive")
	before_mean = sum(sample["scores"]["composite_score"] for sample in before_samples) / runs
	after_mean = sum(sample["scores"]["composite_score"] for sample in after_samples) / runs
	delta = after_mean - before_mean
	observed = "unchanged" if abs(delta) <= tolerance else ("improve" if delta > 0 else "degrade")
	report.update({
		"before_samples": before_samples,
		"after_samples": after_samples,
		"before_mean": before_mean,
		"after_mean": after_mean,
		"score_change": delta,
		"observed": observed,
		"prediction_correct": (
			None if report["prediction"] == "uncertain"
			else report["prediction"] == observed
		),
	})
	return report


if __name__ == "__main__":
	before = {"task": "Explain API rate limits to a beginner."}
	after = {
		**before,
		"role": "A concise API documentation writer",
		"constraints": ["Use at most 100 words", "Include the terms API, requests, and limit"],
		"examples": ['{"explanation": "An API limit caps requests."}'],
		"format": "Only valid JSON, without Markdown fences",
	}
	criteria = {
		"max_words": 100,
		"required_keywords": ["API", "requests", "limit"],
		"expected_format": "json",
	}
	prediction = diff_prompts(before, after)
	print(prediction["text_diff"])
	print(json.dumps(prediction["changes"], indent=2))
	print(f"Heuristic prediction: {prediction['prediction']}. {prediction['reason']}")
	print("Testing with Ollama at temperature=0.7; 5 samples per version, 10 calls.")
	result = test_prediction(before, after, criteria)
	print(f"Before: {result['before_mean']:.3f}; after: {result['after_mean']:.3f}")
	print(f"Observed: {result['observed']} ({result['score_change']:+.3f})")
	print("Prediction supported by this run:", result["prediction_correct"])
	print("Heuristic scoring and a small sample cannot establish general quality or statistical significance.")