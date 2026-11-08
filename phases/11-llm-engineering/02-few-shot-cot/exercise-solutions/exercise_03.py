# Problem Statement:
# **Self-consistency cost curve**:
# Run self-consistency with N=1, 3, 5, 7, 10 on 20 GSM8K
# problems. Plot accuracy vs cost (total tokens). Where is
# the knee of the curve for your model?

import json
import re
from collections import Counter
from itertools import islice
from urllib.request import urlopen

import matplotlib.pyplot as plt
from ollama import Client

MODEL = "llama3.2"
NUM_PROBLEMS = 20
NUM_EXAMPLES = 3
N_VALUES = [1, 3, 5, 7, 10]
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1024
KNEE_TOLERANCE = 5  # Percentage points below the best accuracy

# Official GSM8K repository. Examples come from train, problems from test.
DATA_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/{}.jsonl"
)

SYSTEM = (
    "You are a precise math problem solver. "
    "Show your step-by-step calculations clearly. "
    "End on the last line with exactly: The answer is [number]"
)

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


def build_prompt(question, examples):
    shots = ""
    for row in examples:
        # GSM8K solutions contain calculator annotations like <<2+2=4>>.
        reasoning = re.sub(r"<<[^>]+>>", "", row["answer"].split("####")[0]).strip()
        shots += f"Q: {row['question']}\nA: {reasoning}\nThe answer is {gold_answer(row):g}\n\n"
    return f"{shots}Q: {question}\nA:"


def sample(client, prompt):
    """One sampled response: its answer and the tokens it cost (prompt + output)."""
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": TEMPERATURE, "num_predict": MAX_OUTPUT_TOKENS},
    )
    tokens = response.prompt_eval_count + response.eval_count
    return extract_answer(response.message.content or ""), tokens


def majority_vote(answers):
    """Most common answer. Ties go to whichever answer appeared first."""
    votes = Counter(answer for answer in answers if answer is not None)
    return votes.most_common(1)[0][0] if votes else None


def run_experiment(client, problems, examples):
    """Sample 10 responses per problem once, then reuse the first N for each N."""
    correct = Counter()
    tokens = Counter()
    for i, problem in enumerate(problems, start=1):
        expected = gold_answer(problem)
        prompt = build_prompt(problem["question"], examples)
        samples = [sample(client, prompt) for _ in range(max(N_VALUES))]
        answers = [answer for answer, _ in samples]
        print(f"Problem {i}/{len(problems)} | expected: {expected:g} | samples: {answers}", flush=True)

        for n in N_VALUES:
            correct[n] += majority_vote(answers[:n]) == expected
            tokens[n] += sum(cost for _, cost in samples[:n])
    return correct, tokens


def show_results(accuracy, tokens, knee):
    print(f"\nSELF-CONSISTENCY COST CURVE | Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"{'N':>3} {'Accuracy':>10} {'Total tokens':>14}")
    for n in N_VALUES:
        print(f"{n:>3} {accuracy[n]:>9.1f}% {tokens[n]:>14,}")
    print(
        f"\nKnee: N={knee}, the cheapest N within {KNEE_TOLERANCE} points of the "
        f"best accuracy ({accuracy[knee]:.1f}% at {tokens[knee]:,} tokens)."
    )


def plot_curve(accuracy, tokens, knee):
    costs = [tokens[n] for n in N_VALUES]
    plt.figure(figsize=(9, 5.5))
    plt.plot(costs, [accuracy[n] for n in N_VALUES], marker="o", linewidth=2)
    for n in N_VALUES:
        plt.annotate(f"N={n}", (tokens[n], accuracy[n]), xytext=(6, 8), textcoords="offset points")
    plt.scatter([tokens[knee]], [accuracy[knee]], marker="*", s=180, label=f"Knee: N={knee}")
    plt.xlabel("Total tokens across all problems (prompt + output)")
    plt.ylabel("Majority-vote accuracy (%)")
    plt.title(f"Self-consistency: accuracy vs token cost ({MODEL})")
    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()  # Display interactively. Do NOT save a file.


def main():
    client = Client(host="http://localhost:11434", timeout=180.0)
    problems = load_gsm8k("test", NUM_PROBLEMS)
    examples = load_gsm8k("train", NUM_EXAMPLES)
    correct, tokens = run_experiment(client, problems, examples)

    accuracy = {n: 100 * correct[n] / len(problems) for n in N_VALUES}
    best = max(accuracy.values())
    knee = next(n for n in N_VALUES if accuracy[n] >= best - KNEE_TOLERANCE)

    show_results(accuracy, tokens, knee)
    plot_curve(accuracy, tokens, knee)


if __name__ == "__main__":
    main()
