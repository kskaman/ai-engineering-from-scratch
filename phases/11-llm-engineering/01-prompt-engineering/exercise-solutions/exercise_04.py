# Problem Statement : Implement a prompt optimizer. 
# Given a prompt and a scoring criteria, run the prompt 
# 5 times with temperature=0.7, score each output, identify 
# the weakest criteria, and rewrite the prompt to 
# address it. Repeat for 3 iterations. Measure 
# whether scores improve.

from statistics import mean, pstdev

from exercise_02 import MODEL_CONFIGS, ollama_request, score_response


def request_output(prompt):
	payload = {
		"system": "Follow the user's task and output constraints.",
		"user": prompt,
		"temperature": 0.7,
	}
	result = ollama_request(
		payload, payload["system"], [{"role": "user", "content": prompt}]
	)
	if result["finish_reason"] == "length":
		raise RuntimeError("Output was truncated; increase the Ollama token limit before scoring.")
	if not result["response"].strip():
		raise RuntimeError("The model returned no text; cannot score this sample.")
	return result["response"]


def criterion_instructions(criteria):
	instructions = {}
	if "max_words" in criteria:
		if criteria["max_words"] <= 0:
			raise ValueError("max_words must be positive")
		instructions["length_compliant"] = f"Use at most {criteria['max_words']} words."
	if criteria.get("required_keywords"):
		instructions["keyword_coverage"] = (
			"Include each of these terms explicitly: "
			+ ", ".join(criteria["required_keywords"])
			+ "."
		)
	if criteria.get("forbidden_phrases"):
		instructions["no_violations"] = (
			"Do not use any of these phrases: "
			+ ", ".join(criteria["forbidden_phrases"])
			+ "."
		)
	formats = {
		"json": "Return only valid JSON, with no Markdown fences or surrounding prose.",
		"bullet_points": "Return only a bullet list; start every nonempty line with '- '.",
		"numbered_list": "Return a numbered list with at least two items, starting lines with '1.', '2.', etc.",
	}
	if "expected_format" in criteria:
		if criteria["expected_format"] not in formats:
			raise ValueError("Unsupported expected_format")
		instructions["format_compliant"] = formats[criteria["expected_format"]]
	if not instructions:
		raise ValueError("Provide at least one nonempty scoring criterion")
	return instructions


def evaluate_prompt(prompt, criteria, request=None, runs=5):
	if runs < 1:
		raise ValueError("runs must be positive")
	instructions = criterion_instructions(criteria)
	request = request or request_output
	samples = []
	for _ in range(runs):
		response = request(prompt)
		if not response.strip():
			raise RuntimeError("Cannot evaluate an empty response")
		scores = score_response(response, criteria)
		samples.append({"response": response, "scores": scores})
	averages = {
		criterion: mean(float(sample["scores"][criterion]) for sample in samples)
		for criterion in instructions
	}
	composite_scores = [sample["scores"]["composite_score"] for sample in samples]
	return {
		"prompt": prompt,
		"samples": samples,
		"criterion_averages": averages,
		"mean_score": mean(composite_scores),
		"score_stddev": pstdev(composite_scores),
	}


def optimize_prompt(prompt, criteria, request=None):
	instructions = criterion_instructions(criteria)
	history = [evaluate_prompt(prompt, criteria, request)]
	requirements = {}
	for iteration in range(1, 4):
		previous = history[-1]
		weakest = min(previous["criterion_averages"], key=previous["criterion_averages"].get)
		requirements[weakest] = instructions[weakest]
		revised = prompt + "\n\nOutput requirements:\n" + "\n".join(
			f"- {instruction}" for instruction in requirements.values()
		)
		result = evaluate_prompt(revised, criteria, request)
		result.update({"iteration": iteration, "targeted_criterion": weakest})
		history.append(result)
	return {
		"history": history,
		"best": max(history, key=lambda result: result["mean_score"]),
		"score_change": history[-1]["mean_score"] - history[0]["mean_score"],
	}


if __name__ == "__main__":
	print(f"Ollama model: {MODEL_CONFIGS['ollama']['model']}; temperature=0.7; 20 calls")
	result = optimize_prompt(
		"Explain API rate limits to a beginner.",
		{
			"max_words": 100,
			"required_keywords": ["API", "requests", "limit"],
			"expected_format": "json",
		},
	)
	for iteration, stage in enumerate(result["history"]):
		print(
			f"Stage {iteration}: mean={stage['mean_score']:.3f}, "
			f"stddev={stage['score_stddev']:.3f}, "
			f"target={stage.get('targeted_criterion', 'baseline')}"
		)
		print(f"  Criterion averages: {stage['criterion_averages']}")
	print(f"Final minus baseline: {result['score_change']:+.3f}")
	print("Improvement observed:", result["score_change"] > 0)
	print("Best measured prompt:\n" + result["best"]["prompt"])
	print("These heuristic scores and five samples per stage do not establish statistical significance.")