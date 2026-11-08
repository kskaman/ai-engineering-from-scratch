# Problem Statement:
# Build a ReAct loop: Extend the pipeline with a calculator
# tool. When the model generates a math expression, execute
#  it with Python's eval() (in a sandbox) and feed the
# result back. Measure if tool-grounded reasoning outperforms pure CoT.

import json
import re
from itertools import islice
from urllib.request import urlopen

from ollama import Client

MODEL = "llama3.2"
NUM_PROBLEMS = 20
MAX_STEPS = 10
MAX_OUTPUT_TOKENS = 1024  # Long enough that CoT isn't cut off before its answer

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)

# Worked examples from the GSM8K train split. Both prompts are rendered from the
# same steps, so the calculator is the only difference between the two methods.
EXAMPLES = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May?"
        ),
        "steps": [
            ("May sales are half of April sales.", "48 / 2", "24"),
            ("Add the sales of both months.", "48 + 24", "72"),
        ],
        "answer": "72",
    },
    {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
            "minutes of babysitting. How much did she earn?"
        ),
        "steps": [
            ("Find her pay per minute.", "12 / 60", "0.2"),
            ("Multiply by the minutes she worked.", "0.2 * 50", "10"),
        ],
        "answer": "10",
    },
    {
        "question": (
            "Betty is saving money for a new wallet which costs $100. Betty has only "
            "half of the money she needs. Her parents decided to give her $15 for "
            "that purpose, and her grandparents twice as much as her parents. How "
            "much more money does Betty need to buy the wallet?"
        ),
        "steps": [
            ("Betty already has half of the price.", "100 / 2", "50"),
            ("Her grandparents give twice what her parents give.", "15 * 2", "30"),
            ("Add all the money she has.", "50 + 15 + 30", "95"),
            ("Subtract that from the price.", "100 - 95", "5"),
        ],
        "answer": "5",
    },
]

NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
ANSWER_RE = re.compile(rf"answer(?:\s+is)?[\s:*$]*({NUMBER})", re.IGNORECASE)
ACTION_RE = re.compile(r"Action:\s*calculate\s*(.+)", re.IGNORECASE)
ARITHMETIC_RE = re.compile(r"[\d\s+\-*/().]+")


def to_number(text):
    return float(text.replace(",", ""))


def extract_answer(text):
    matches = ANSWER_RE.findall(text)
    return to_number(matches[-1]) if matches else None


def calculate(expression):
    """Sandboxed eval(): only arithmetic reaches it, and it has no builtins."""
    expression = expression.split("=")[0].replace(",", "").replace("$", "").strip()
    # "**" is blocked because something like 9**9**9 would hang the program.
    if not ARITHMETIC_RE.fullmatch(expression) or "**" in expression:
        return "Error: use only numbers and + - * / ( ). Write percents as decimals."
    try:
        return str(round(eval(expression, {"__builtins__": {}}, {}), 6))
    except Exception as exc:
        return f"Error: {exc}"


def chat(client, messages, stop=None):
    response = client.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": 0.0, "num_predict": MAX_OUTPUT_TOKENS, "stop": stop},
    )
    return response.message.content or ""


def cot_example(example):
    steps = " ".join(f"{why} {expr} = {result}." for why, expr, result in example["steps"])
    return f"Q: {example['question']}\nA: {steps} The answer is {example['answer']}."


def react_example(example):
    lines = [f"Q: {example['question']}"]
    for why, expr, result in example["steps"]:
        lines += [f"Thought: {why}", f"Action: calculate {expr}", f"Observation: {result}"]
    lines.append(f"Answer: {example['answer']}")
    return "\n".join(lines)


def cot_solve(client, question):
    """Pure CoT: the model does all of the arithmetic in its own head."""
    system = (
        "You are a precise math problem solver. "
        "Show your step-by-step reasoning. "
        "End on the last line with exactly: The answer is [number]"
    )
    examples = "\n\n".join(cot_example(example) for example in EXAMPLES)
    text = chat(client, [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{examples}\n\nQ: {question}\nA:"},
    ])
    return extract_answer(text)


def react_solve(client, question):
    """ReAct: Thought -> Action -> Observation, until the model gives an Answer."""
    examples = "\n\n".join(react_example(example) for example in EXAMPLES)
    system = (
        "You are a math problem solver with a calculator. "
        "Never do arithmetic yourself. Reply with one step at a time:\n"
        "Thought: [your reasoning]\n"
        "Action: calculate [expression]\n"
        "You will then receive the result as an Observation. "
        "When you have the final number, reply with:\n"
        "Answer: [number]\n\n"
        f"Examples:\n\n{examples}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Q: {question}"},
    ]

    tool_calls = 0
    for _ in range(MAX_STEPS):
        # The stop sequence prevents the model from inventing its own Observation.
        text = chat(client, messages, stop=["Observation:"])

        action = ACTION_RE.search(text)
        if action:
            tool_calls += 1
            messages += [
                {"role": "assistant", "content": text[:action.end()]},
                {"role": "user", "content": f"Observation: {calculate(action.group(1))}"},
            ]
            continue

        answer = extract_answer(text)
        if answer is not None:
            return answer, tool_calls

        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Reply with an Action or the final Answer."},
        ]

    return None, tool_calls


def load_problems():
    with urlopen(GSM8K_TEST_URL, timeout=30) as response:
        return [json.loads(line) for line in islice(response, NUM_PROBLEMS)]


def run_comparison(client, problems):
    rows = []
    for i, problem in enumerate(problems, start=1):
        expected = to_number(problem["answer"].split("####")[1].strip())
        cot_answer = cot_solve(client, problem["question"])
        react_answer, tool_calls = react_solve(client, problem["question"])
        rows.append((cot_answer == expected, react_answer == expected, tool_calls))
        print(
            f"Problem {i:2d}/{len(problems)} | expected: {expected:g} | "
            f"CoT: {cot_answer} | ReAct: {react_answer} ({tool_calls} calculator calls)",
            flush=True,
        )
    return rows


def show_results(rows):
    total = len(rows)
    cot_correct = sum(cot for cot, _, _ in rows)
    react_correct = sum(react for _, react, _ in rows)
    only_cot = sum(cot and not react for cot, react, _ in rows)
    only_react = sum(react and not cot for cot, react, _ in rows)
    average_calls = sum(calls for _, _, calls in rows) / total

    print(f"\nRESULTS | Model: {MODEL} | Problems: {total}")
    print(f"  Pure CoT accuracy: {cot_correct}/{total} ({100 * cot_correct / total:.1f}%)")
    print(f"  ReAct accuracy:    {react_correct}/{total} ({100 * react_correct / total:.1f}%)")
    print(f"  Only CoT correct:   {only_cot}")
    print(f"  Only ReAct correct: {only_react}")
    print(f"  Average calculator calls per problem: {average_calls:.1f}")

    if react_correct > cot_correct:
        verdict = "tool-grounded ReAct outperformed pure CoT"
    elif react_correct < cot_correct:
        verdict = "tool-grounded ReAct did NOT outperform pure CoT"
    else:
        verdict = "tool-grounded ReAct and pure CoT tied"
    print(f"\nVerdict: {verdict}.")
    print(f"One problem is worth {100 / total:.1f} points, so treat small gaps as noise.")


def main():
    client = Client(host="http://localhost:11434", timeout=180.0)
    rows = run_comparison(client, load_problems())
    show_results(rows)


if __name__ == "__main__":
    main()
