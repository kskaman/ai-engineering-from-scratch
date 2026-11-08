# Problem Statement:
# **Example selection experiment:**
#   For the same 10 problems, compare random example
#   selection vs hand-picked similar examples. Measure
#   accuracy difference. At what point does example quality
#   matter more than example quantity?

import json
import random
import re
from itertools import islice
from statistics import mean, stdev
from urllib.request import urlopen

from ollama import Client

MODEL = "llama3.2"
NUM_PROBLEMS = 10
POOL_SIZE = 60
K_VALUES = (1, 2, 4, 8)
RANDOM_TRIALS = 5
MAX_OUTPUT_TOKENS = 1024
SEED = 42

# Official GSM8K repository. Examples come from train, problems from test.
DATA_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/{}.jsonl"
)

SYSTEM = (
    "You are a precise math problem solver. "
    "Solve the problem step by step. "
    "End on the last line with exactly: The answer is [number]"
)

# For each test problem, the pool rows with the most similar reasoning, best first.
# Chosen by reasoning structure only, before any example-selection run.
HAND_PICKED = [
    [44, 23, 54, 28, 8, 17, 31, 2],     # Eggs: subtract from a total, then multiply
    [0, 34, 33, 12, 20, 36, 6, 46],     # Robe: "half as much", then a total
    [15, 40, 5, 29, 16, 27, 44, 31],    # House: percent increase, then profit
    [4, 21, 46, 18, 19, 6, 39, 14],     # Sprints: chain of multiplications
    [56, 28, 23, 8, 24, 17, 59, 2],     # Feed: total needed minus parts already given
    [40, 9, 13, 42, 37, 47, 45, 53],    # Glasses: two price tiers, then a total
    [49, 20, 12, 34, 5, 21, 48, 36],    # Sheep: chain of "times as many", then a total
    [47, 27, 19, 14, 26, 18, 56, 35],   # Download: percent of a file, rate, added time
    [19, 47, 32, 18, 14, 9, 56, 28],    # Drive: speed x time across several phases
    [9, 40, 47, 1, 11, 50, 31, 58],     # Pay: overtime above an hours threshold
]

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


def format_example(row):
    # GSM8K solutions contain calculator annotations like <<2+2=4>>.
    reasoning = re.sub(r"<<[^>]+>>", "", row["answer"].split("####")[0]).strip()
    return f"Q: {row['question']}\nA: {reasoning}\nThe answer is {gold_answer(row):g}\n\n"


def solve(client, question, examples):
    shots = "".join(format_example(row) for row in examples)
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{shots}Q: {question}\nA:"},
        ],
        options={"temperature": 0.0, "num_predict": MAX_OUTPUT_TOKENS},
    )
    return extract_answer(response.message.content or "")


def count_correct(client, problems, example_sets):
    """Solve every problem, each with its own set of examples."""
    return sum(
        solve(client, problem["question"], examples) == gold_answer(problem)
        for problem, examples in zip(problems, example_sets)
    )


def run_hand_picked(client, problems, pool):
    scores = {}
    for k in K_VALUES:
        example_sets = [[pool[i] for i in ranking[:k]] for ranking in HAND_PICKED]
        scores[k] = count_correct(client, problems, example_sets)
        print(f"Hand-picked k={k}: {scores[k]}/{len(problems)}", flush=True)
    return scores


def run_random(client, problems, pool):
    rng = random.Random(SEED)
    scores = {k: [] for k in K_VALUES}
    for trial in range(1, RANDOM_TRIALS + 1):
        # One draw per problem. Every k uses the start of the same draw, so
        # differences between k values come from quantity, not a luckier draw.
        draws = [rng.sample(pool, max(K_VALUES)) for _ in problems]
        for k in K_VALUES:
            scores[k].append(count_correct(client, problems, [draw[:k] for draw in draws]))
        latest = ", ".join(f"k={k}: {scores[k][-1]}" for k in K_VALUES)
        print(f"Random trial {trial}/{RANDOM_TRIALS}: {latest}", flush=True)
    return scores


def show_results(hand, random_scores, total):
    print(f"\nACCURACY | Model: {MODEL} | Problems: {total} | Random trials: {RANDOM_TRIALS}")
    print(f"{'k':>3} {'Random mean':>12} {'Random SD':>10} {'Hand-picked':>12} {'Gap':>8}")
    for k in K_VALUES:
        random_mean = 100 * mean(random_scores[k]) / total
        random_sd = 100 * stdev(random_scores[k]) / total
        hand_accuracy = 100 * hand[k] / total
        print(
            f"{k:>3} {random_mean:>11.1f}% {random_sd:>9.1f}% "
            f"{hand_accuracy:>11.1f}% {hand_accuracy - random_mean:>+7.1f}%"
        )

    # Quality beats quantity when fewer hand-picked examples beat more random ones.
    print("\nQuality vs quantity:")
    wins = [
        (small, large) for small in K_VALUES for large in K_VALUES
        if small < large and hand[small] > mean(random_scores[large])
    ]
    for small, large in wins:
        print(f"  Hand-picked k={small} beat random k={large}")
    if not wins:
        print("  No hand-picked set beat a larger random set.")
    print(f"\nOne problem is worth {100 / total:.0f} points, so treat small gaps as noise.")


def main():
    client = Client(host="http://localhost:11434", timeout=180.0)
    problems = load_gsm8k("test", NUM_PROBLEMS)
    pool = load_gsm8k("train", POOL_SIZE)
    hand = run_hand_picked(client, problems, pool)
    random_scores = run_random(client, problems, pool)
    show_results(hand, random_scores, len(problems))


if __name__ == "__main__":
    main()
