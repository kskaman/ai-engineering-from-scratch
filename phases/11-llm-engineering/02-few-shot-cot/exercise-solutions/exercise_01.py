# Problem Statement:
#   **Measure the gap**:
#   Take 10 GSM8K problems. Solve each with zero-shot,
#   few-shot, zero-shot CoT, and few-shot CoT. Record
#   accuracy for each. Which technique gives the biggest
#   lift on your model?

import json
import re
from itertools import islice
from urllib.request import urlopen

from ollama import Client

MODEL = "llama3.2"
NUM_PROBLEMS = 10
NUM_EXAMPLES = 3
MAX_OUTPUT_TOKENS = 1024

# Official GSM8K repository. Examples come from train, problems from test.
DATA_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/{}.jsonl"
)

# Technique name: (uses examples, uses reasoning)
TECHNIQUES = {
    "zero-shot": (False, False),
    "few-shot": (True, False),
    "zero-shot CoT": (False, True),
    "few-shot CoT": (True, True),
}

DIRECT_SYSTEM = (
    "You are a precise math problem solver. "
    "Give only the final numerical answer."
)
COT_SYSTEM = (
    "You are a precise math problem solver. "
    "Show your step-by-step reasoning. "
    "End on the last line with exactly: The answer is [number]"
)

# Without this schema, the model often reasons anyway on "answer only" prompts,
# which would hide part of the lift that CoT gives.
ANSWER_ONLY = {
    "type": "object",
    "properties": {"answer": {"type": "number"}},
    "required": ["answer"],
}

NUMBER = r"\d[\d,]*(?:\.\d+)?"
MARKED_ANSWER = re.compile(rf"answer(?:\s+is)?[\s:*$\"]*(-?{NUMBER})", re.IGNORECASE)


def load_gsm8k(split, count):
    with urlopen(DATA_URL.format(split), timeout=30) as response:
        return [json.loads(line) for line in islice(response, count)]


def to_number(text):
    return float(text.replace(",", ""))


def gold_answer(row):
    return to_number(row["answer"].split("####")[1].strip())


def extract_answer(text):
    """Use the last marked answer, otherwise the last number in the text."""
    matches = MARKED_ANSWER.findall(text) or re.findall(NUMBER, text)
    return to_number(matches[-1]) if matches else None


def format_example(row, use_cot):
    # GSM8K solutions contain calculator annotations like <<2+2=4>>.
    reasoning = re.sub(r"<<[^>]+>>", "", row["answer"].split("####")[0]).strip()
    worked = f"{reasoning}\n" if use_cot else ""
    return f"Q: {row['question']}\nA: {worked}The answer is {gold_answer(row):g}.\n\n"


def build_prompt(question, examples, use_cot):
    shots = "".join(format_example(row, use_cot) for row in examples)
    # Few-shot CoT learns to reason from its examples. Zero-shot CoT needs this cue.
    cue = " Let's think step by step." if use_cot and not examples else ""
    return f"{shots}Q: {question}\nA:{cue}"


def solve(client, question, examples, use_cot):
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": COT_SYSTEM if use_cot else DIRECT_SYSTEM},
            {"role": "user", "content": build_prompt(question, examples, use_cot)},
        ],
        format=None if use_cot else ANSWER_ONLY,
        options={"temperature": 0.0, "num_predict": MAX_OUTPUT_TOKENS},
    )
    return extract_answer(response.message.content or "")


def run_comparison(client, problems, examples):
    correct = {name: 0 for name in TECHNIQUES}
    for i, problem in enumerate(problems, start=1):
        expected = gold_answer(problem)
        print(f"\nProblem {i}/{len(problems)} | expected: {expected:g}", flush=True)
        for name, (use_examples, use_cot) in TECHNIQUES.items():
            shots = examples if use_examples else []
            answer = solve(client, problem["question"], shots, use_cot)
            correct[name] += answer == expected
            print(f"  {name:14s}: {answer}", flush=True)
    return correct


def show_results(correct, total):
    print(f"\nACCURACY | Model: {MODEL} | Problems: {total}")
    for name, score in correct.items():
        print(f"  {name:14s}: {score}/{total} ({100 * score / total:.0f}%)")

    baseline = correct["zero-shot"]
    lifts = {name: score - baseline for name, score in correct.items() if name != "zero-shot"}
    best = max(lifts.values())
    winners = [name for name, lift in lifts.items() if lift == best]
    print(f"\nBiggest lift over zero-shot: {', '.join(winners)} ({best:+d} problems)")


def main():
    client = Client(host="http://localhost:11434", timeout=180.0)
    problems = load_gsm8k("test", NUM_PROBLEMS)
    examples = load_gsm8k("train", NUM_EXAMPLES)
    correct = run_comparison(client, problems, examples)
    show_results(correct, len(problems))


if __name__ == "__main__":
    main()
