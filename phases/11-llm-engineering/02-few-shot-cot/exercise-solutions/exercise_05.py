# Problem Statement:
# ToT for creative tasks: Adapt the Tree-of-Thought
# solver for a creative writing task: "Write a 6-word
# story that is both funny and sad." Use the LLM as
# evaluator. Does branching exploration produce better
# creative outputs than single-shot generation?

import json
import random
import re
from collections import Counter

from ollama import Client

MODEL = "llama3.2"
TASK = "Write a 6-word story that is both funny and sad."
ROUNDS = 10
BREADTH = 3  # Branches created at each node
KEEP = 2     # Ideas that survive pruning
THEMES = [
    "a pet", "a birthday", "a first date", "old age",
    "a job", "a family dinner", "technology", "moving house",
]

# Every generated node costs one call and is scored by one more call.
TOT_MAX_CALLS = 2 * BREADTH + 2 * KEEP * BREADTH

TEXT = {"type": "string"}
SCORE = {"type": "integer"}
WINNER = {"type": "string", "enum": ["A", "B"]}


def ask(client, prompt, temperature, **fields):
    """Call the LLM and get JSON back containing exactly the requested fields."""
    schema = {"type": "object", "properties": fields, "required": list(fields)}
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": temperature, "num_predict": 200},
    )
    return json.loads(response.message.content)


def is_six_words(story):
    return len(re.findall(r"\w[\w'’-]*", story)) == 6


def generate_idea(client, theme):
    prompt = (
        f"{TASK}\nDo not write the story yet. "
        f"Describe one premise for it in one sentence. Theme: {theme}."
    )
    return ask(client, prompt, 0.9, premise=TEXT)["premise"]


def write_story(client, idea=None):
    """Write one story. Without an idea, this is plain single-shot generation."""
    premise = f"\nBase it on this premise: {idea}" if idea else ""
    prompt = f"{TASK}{premise}\nThe story must have exactly 6 words."
    return ask(client, prompt, 0.9, story=TEXT)["story"]


def evaluate(client, text):
    """LLM evaluator. The weakest score counts because it must be funny AND sad."""
    prompt = (
        "You are a strict critic judging material for a 6-word story that must be "
        "both funny and sad.\n"
        "Score how funny and how sad this is, from 0 (not at all) to 10 (brilliant).\n\n"
        f"Text: {text}"
    )
    scores = ask(client, prompt, 0.0, funny=SCORE, sad=SCORE)
    return min(scores["funny"], scores["sad"], 10)


def evaluate_story(client, story):
    # Models can't count words reliably, so code enforces the 6-word rule.
    return evaluate(client, story) if is_six_words(story) else -1


def tree_of_thought(client):
    """Branch into ideas, prune to the best, branch into stories, pick the best."""
    themes = random.sample(THEMES, BREADTH)
    ideas = [generate_idea(client, theme) for theme in themes]
    best_ideas = sorted(ideas, key=lambda idea: evaluate(client, idea), reverse=True)[:KEEP]
    stories = [write_story(client, idea) for idea in best_ideas for _ in range(BREADTH)]
    return max(stories, key=lambda story: evaluate_story(client, story))


def judge(client, story_a, story_b):
    prompt = (
        "Which 6-word story is better at being both funny and sad?\n"
        f"A: {story_a}\nB: {story_b}"
    )
    return ask(client, prompt, 0.0, winner=WINNER)["winner"]


def compare(client, tot_story, single_story):
    """A fresh judge decides, as ToT's winner already maximized the evaluator's score."""
    tot_valid, single_valid = is_six_words(tot_story), is_six_words(single_story)
    if not tot_valid and not single_valid:
        return "Both invalid"
    if not single_valid:
        return "ToT by forfeit"
    if not tot_valid:
        return "Single-shot by forfeit"

    # Ask twice with positions swapped, so a bias toward A or B can't pick the winner.
    first = judge(client, tot_story, single_story)
    second = judge(client, single_story, tot_story)
    if first == "A" and second == "B":
        return "ToT"
    if first == "B" and second == "A":
        return "Single-shot"
    return "Tie"


def run_rounds(client):
    outcomes = Counter()
    valid = Counter()
    for i in range(1, ROUNDS + 1):
        single_story = write_story(client)
        tot_story = tree_of_thought(client)
        outcome = compare(client, tot_story, single_story)

        outcomes[outcome] += 1
        valid["Single-shot"] += is_six_words(single_story)
        valid["ToT"] += is_six_words(tot_story)
        print(
            f"Round {i}/{ROUNDS}\n"
            f"  Single-shot: {single_story}\n"
            f"  ToT:         {tot_story}\n"
            f"  Winner:      {outcome}",
            flush=True,
        )
    return outcomes, valid


def show_results(outcomes, valid):
    tot_wins, single_wins = outcomes["ToT"], outcomes["Single-shot"]

    print(f"\nRESULTS | Model: {MODEL} | Rounds: {ROUNDS}")
    print(f"  Exactly 6 words:  ToT {valid['ToT']}/{ROUNDS}, Single-shot {valid['Single-shot']}/{ROUNDS}")
    print(f"  Judged by LLM:    ToT {tot_wins}, Single-shot {single_wins}, Tie {outcomes['Tie']}")
    print(
        f"  Won by forfeit:   ToT {outcomes['ToT by forfeit']}, "
        f"Single-shot {outcomes['Single-shot by forfeit']}, "
        f"Both invalid {outcomes['Both invalid']}"
    )
    print(f"  Calls per story:  ToT up to {TOT_MAX_CALLS}, Single-shot 1")

    if tot_wins > single_wins:
        verdict = "branching produced better stories, according to the LLM judge"
    elif tot_wins < single_wins:
        verdict = "branching did NOT produce better stories, according to the LLM judge"
    else:
        verdict = "branching and single-shot tied, according to the LLM judge"
    print(f"\nVerdict: {verdict}.")


def main():
    client = Client(host="http://localhost:11434", timeout=180.0)
    outcomes, valid = run_rounds(client)
    show_results(outcomes, valid)


if __name__ == "__main__":
    main()
