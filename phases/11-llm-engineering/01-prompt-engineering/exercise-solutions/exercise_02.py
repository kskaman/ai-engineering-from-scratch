# Problem Statement : Replace simulate_llm_call with real 
# API calls to at least two providers (OpenAI and 
# Anthropic free tiers work). Run the same prompt 
# across both and measure: response length, format 
# compliance, keyword coverage, and latency. Document 
# which model follows instructions more precisely.


import json
import time
import hashlib
import re
import os

from dotenv import load_dotenv

load_dotenv()

PROMPT_PATTERNS = {
    "persona": {
        "name": "Persona Pattern",
        "template": (
            "You are {role} with {experience}.\n"
            "Your communication style is {style}.\n"
            "You prioritize {priority}.\n\n"
            "{task}"
        ),
        "variables": ["role", "experience", "style", "priority", "task"],
        "temperature": 0.7,
        "description": "Activates a specific expert distribution in the model's training data",
    },
    "few_shot": {
        "name": "Few-Shot Pattern",
        "template": (
            "Here are examples of the expected input/output format:\n\n"
            "{examples}\n\n"
            "Now process this input:\n{input}"
        ),
        "variables": ["examples", "input"],
        "temperature": 0.0,
        "description": "Provides concrete examples to anchor the output format and style",
    },
    "chain_of_thought": {
        "name": "Chain-of-Thought Pattern",
        "template": (
            "Think through this step by step.\n\n"
            "Problem: {problem}\n\n"
            "Steps:\n"
            "1. Identify the key components\n"
            "2. Analyze each component\n"
            "3. Synthesize your findings\n"
            "4. State your conclusion\n\n"
            "Show your reasoning before giving the final answer."
        ),
        "variables": ["problem"],
        "temperature": 0.3,
        "description": "Forces explicit reasoning steps before the final answer",
    },
    "template_fill": {
        "name": "Template Fill Pattern",
        "template": (
            "Extract information from the following text and fill in the template.\n\n"
            "Text: {text}\n\n"
            "Template:\n{template_structure}\n\n"
            "Fill in every field. If information is not available, write 'N/A'."
        ),
        "variables": ["text", "template_structure"],
        "temperature": 0.0,
        "description": "Constrains output to a specific structure with named fields",
    },
    "critique": {
        "name": "Critique Pattern",
        "template": (
            "Task: {task}\n\n"
            "Step 1: Generate an initial response.\n"
            "Step 2: Critique your response for accuracy, completeness, and clarity.\n"
            "Step 3: Produce an improved final version.\n\n"
            "Label each step clearly."
        ),
        "variables": ["task"],
        "temperature": 0.5,
        "description": "Self-refinement through explicit critique before final output",
    },
    "guardrail": {
        "name": "Guardrail Pattern",
        "template": (
            "You are a {role}.\n\n"
            "Rules:\n"
            "- ONLY answer questions about {domain}\n"
            "- If the question is outside {domain}, say: 'This is outside my scope.'\n"
            "- NEVER make up information. If unsure, say 'I don't know.'\n"
            "- {additional_rules}\n\n"
            "User question: {question}"
        ),
        "variables": ["role", "domain", "additional_rules", "question"],
        "temperature": 0.3,
        "description": "Constrains the model to a specific domain with explicit boundaries",
    },
    "meta_prompt": {
        "name": "Meta-Prompt Pattern",
        "template": (
            "Write a prompt for an LLM that will {objective}.\n\n"
            "The prompt should include:\n"
            "- A specific role/persona\n"
            "- Clear constraints and output format\n"
            "- 2-3 few-shot examples\n"
            "- Edge case handling\n\n"
            "Optimize the prompt for {metric}.\n"
            "Target model: {model}."
        ),
        "variables": ["objective", "metric", "model"],
        "temperature": 0.7,
        "description": "Uses the LLM to generate optimized prompts for other tasks",
    },
    "decomposition": {
        "name": "Decomposition Pattern",
        "template": (
            "Problem: {problem}\n\n"
            "Break this into sub-problems:\n"
            "1. List each sub-problem\n"
            "2. Solve each independently\n"
            "3. Combine sub-solutions into a final answer\n"
            "4. Verify the final answer against the original problem"
        ),
        "variables": ["problem"],
        "temperature": 0.3,
        "description": "Breaks complex problems into manageable pieces",
    },
    "audience_adapt": {
        "name": "Audience Adaptation Pattern",
        "template": (
            "Explain {concept} for the following audience: {audience}.\n\n"
            "Constraints:\n"
            "- Use vocabulary appropriate for {audience}\n"
            "- Length: {length}\n"
            "- Include {include}\n"
            "- Exclude {exclude}"
        ),
        "variables": ["concept", "audience", "length", "include", "exclude"],
        "temperature": 0.5,
        "description": "Adapts explanation complexity to the target audience",
    },
    "boundary": {
        "name": "Boundary Pattern",
        "template": (
            "You are an assistant that ONLY handles {scope}.\n\n"
            "If the user's request is within scope, help them fully.\n"
            "If the user's request is outside scope, respond exactly with:\n"
            "'{refusal_message}'\n\n"
            "Do not attempt to answer out-of-scope questions.\n\n"
            "User: {user_input}"
        ),
        "variables": ["scope", "refusal_message", "user_input"],
        "temperature": 0.0,
        "description": "Hard boundary on what the model will and will not respond to",
    },
}




def build_prompt(pattern_name, variables, system_override=None):
    pattern = PROMPT_PATTERNS.get(pattern_name)
    if not pattern:
        raise ValueError(f"Unknown pattern: {pattern_name}. Available: {list(PROMPT_PATTERNS.keys())}")

    missing = [v for v in pattern["variables"] if v not in variables]
    if missing:
        raise ValueError(f"Missing variables for {pattern_name}: {missing}")

    rendered = pattern["template"].format(**variables)
    system = system_override or f"You are an AI assistant using the {pattern['name']}."

    return {
        "system": system,
        "user": rendered,
        "temperature": pattern["temperature"],
        "pattern": pattern_name,
        "metadata": {
            "description": pattern["description"],
            "variables_used": list(variables.keys()),
        },
    }


def build_multi_turn(pattern_name, turns, system_override=None):
    pattern = PROMPT_PATTERNS.get(pattern_name)
    if not pattern:
        raise ValueError(f"Unknown pattern: {pattern_name}")

    system = system_override or f"You are an AI assistant using the {pattern['name']}."
    messages = [{"role": "system", "content": system}]
    for role, content in turns:
        messages.append({"role": role, "content": content})

    return {
        "messages": messages,
        "temperature": pattern["temperature"],
        "pattern": pattern_name,
    }



MODEL_CONFIGS = {
    "openai": {
        "model": os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        "max_tokens": 2048,
    },
    "anthropic": {
        # This model accepts pattern-specific temperature values.
        # Newer Claude 5 models generally reject non-default temperature values.
        "model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        "max_tokens": 2048,
    },
    "ollama": {
        "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
        "max_tokens": 2048,
    },
}


def openai_request(prompt, system, messages):
    model = MODEL_CONFIGS["openai"]["model"]
    max_tokens = MODEL_CONFIGS["openai"]["max_tokens"]
    temperature = prompt["temperature"]

    from openai import OpenAI
    
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing from .env")
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=90, max_retries=1)
    
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *messages],
        "max_completion_tokens": max_tokens,
    }

    if not model.startswith(("gpt-6", "o1", "o3", "o4")):
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    
    usage = response.usage
    
    return {
        "response": response.choices[0].message.content or "",
        "tokens_used": {
            "prompt": usage.prompt_tokens if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        },
        "finish_reason": response.choices[0].finish_reason,
    }


def anthropic_request(prompt, system, messages):

    model = MODEL_CONFIGS["anthropic"]["model"]
    max_tokens = MODEL_CONFIGS["anthropic"]["max_tokens"]
    temperature = prompt["temperature"]

    
    from anthropic import Anthropic
        
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing from .env")
        
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=90, max_retries=1)
        
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    
    # As of Claude 5, non-default temperature isn't accepted. Keep the
    # per-pattern setting on the 4.6 family, via extra_body for newer SDKs.
    if model.startswith(("claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5")):
        kwargs["extra_body"] = {"temperature": temperature}
        
    response = client.messages.create(**kwargs)
        
    output_text = "\n".join(
            block.text for block in response.content if block.type == "text"
    )
        
    usage = response.usage
        
    prompt_tokens = (
        usage.input_tokens
        + (getattr(usage, "cache_creation_input_tokens", None) or 0)
        + (getattr(usage, "cache_read_input_tokens", None) or 0)
    )
    
    return {
        "response": output_text,
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": usage.output_tokens,
            "total": prompt_tokens + usage.output_tokens,
        },
        "finish_reason": response.stop_reason,
    }

def ollama_request(prompt, system, messages):
    model = MODEL_CONFIGS["ollama"]["model"]
    max_tokens = MODEL_CONFIGS["ollama"]["max_tokens"]
    temperature = prompt["temperature"]

    from ollama import Client
    
    client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"), timeout=120)
    
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        options={"temperature": temperature, "num_predict": max_tokens},
        stream=False,
    )
        
    prompt_tokens = response.prompt_eval_count or 0
    completion_tokens = response.eval_count or 0
    
    return {
        "response": response.message.content or "",
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "finish_reason": response.done_reason or "done",
    }

def split_prompt(prompt):
    """Support both single-turn build_prompt and build_multi_turn payloads."""
    if "messages" in prompt:
        messages = prompt["messages"]
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat_messages = [m for m in messages if m["role"] != "system"]
        return "\n\n".join(system_parts), chat_messages
    return prompt["system"], [{"role": "user", "content": prompt["user"]}]


def run_prompt(prompt):
       
    providers = list(MODEL_CONFIGS.keys())

    results = {}
    
    for provider in providers:

        system, messages = split_prompt(prompt)

        start = time.time()

        if provider == "openai":
            print("Running OpenAI request...")
            response = openai_request(prompt, system, messages)
            print("OpenAI request completed.")
        elif provider == "anthropic":
            print("Running Anthropic request...")
            response = anthropic_request(prompt, system, messages)
            print("Anthropic request completed.")
        elif provider == "ollama":
            print("Running Ollama request...")
            response = ollama_request(prompt, system, messages)
            print("Ollama request completed.")

        wall_time = (time.time() - start) * 1000

        results[provider] = {
            "response": response["response"],
            "tokens": response["tokens_used"],
            "api_latency_ms": round(wall_time, 1),
            "wall_time_ms": round(wall_time, 1),
            "finish_reason": response.get("finish_reason"),
        }

    return results


def score_response(response_text, criteria):
    scores = {
        "response_chars": len(response_text),
        "response_words": len(response_text.split()),
        "format_compliant": None,
        "keyword_coverage": None,
    }

    if "max_words" in criteria:
        word_count = len(response_text.split())
        scores["word_count"] = word_count
        scores["length_compliant"] = word_count <= criteria["max_words"]

    if "required_keywords" in criteria:
        found = [kw for kw in criteria["required_keywords"] if kw.lower() in response_text.lower()]
        scores["keywords_found"] = found
        scores["keyword_coverage"] = (
            len(found) / len(criteria["required_keywords"])
            if criteria["required_keywords"]
            else 1.0
        )

    if "forbidden_phrases" in criteria:
        violations = [fp for fp in criteria["forbidden_phrases"] if fp.lower() in response_text.lower()]
        scores["forbidden_violations"] = violations
        scores["no_violations"] = len(violations) == 0

    if "expected_format" in criteria:
        fmt = criteria["expected_format"]
        if fmt == "json":
            try:
                json.loads(response_text)
                scores["format_compliant"] = True
            except (json.JSONDecodeError, TypeError):
                scores["format_compliant"] = False
        elif fmt == "bullet_points":
            lines = [line.strip() for line in response_text.split("\n") if line.strip()]
            bullet_lines = [line for line in lines if line.startswith(("-", "*", "1"))]
            scores["format_compliant"] = len(bullet_lines) >= len(lines) * 0.5
        elif fmt == "numbered_list":
            numbered = re.findall(r"^\d+\.", response_text, re.MULTILINE)
            scores["format_compliant"] = len(numbered) >= 2
        else:
            scores["format_compliant"] = True

    total = 0
    count = 0
    for key, value in scores.items():
        if isinstance(value, bool):
            total += 1.0 if value else 0.0
            count += 1
        elif isinstance(value, float) and 0 <= value <= 1:
            total += value
            count += 1

    scores["composite_score"] = round(total / count, 3) if count > 0 else 0.0
    return scores


def compare_models(test_results, criteria):
    comparison = {}
    for provider, result in test_results.items():
        scores = score_response(result["response"], criteria)
        comparison[provider] = {
            "model": MODEL_CONFIGS[provider]["model"],
            "scores": scores,
            "tokens": result["tokens"],
            "latency_ms": result["api_latency_ms"],
        }

    ranked = sorted(
        comparison.items(),
        key=lambda x: x[1]["scores"]["composite_score"],
        reverse=True,
    )
    return comparison, ranked


TEST_SUITE = [
    {
        "name": "Persona: Technical Writer",
        "pattern": "persona",
        "variables": {
            "role": "a senior technical writer at Stripe",
            "experience": "10 years of API documentation experience",
            "style": "precise, concise, and example-driven",
            "priority": "clarity over comprehensiveness",
            "task": "Explain what an API rate limit is and why it exists.",
        },
        "criteria": {
            "max_words": 200,
            "required_keywords": ["rate limit", "API", "requests"],
            "forbidden_phrases": ["in conclusion", "it is important to note"],
        },
    },
    {
        "name": "Few-Shot: Sentiment Analysis",
        "pattern": "few_shot",
        "variables": {
            "examples": (
                'Input: "The food was amazing but service was slow"\n'
                'Output: {"sentiment": "mixed", "food": "positive", "service": "negative"}\n\n'
                'Input: "Terrible experience, never coming back"\n'
                'Output: {"sentiment": "negative", "food": null, "service": "negative"}'
            ),
            "input": "Great ambiance and the pasta was perfect, though a bit pricey",
        },
        "criteria": {
            "expected_format": "json",
            "required_keywords": ["sentiment"],
        },
    },
    {
        "name": "Chain-of-Thought: Math Problem",
        "pattern": "chain_of_thought",
        "variables": {
            "problem": (
                "A store offers 20% off all items. An item originally costs $85. "
                "There is also a $10 coupon. Which saves more: applying the discount "
                "first then the coupon, or the coupon first then the discount?"
            ),
        },
        "criteria": {
            "required_keywords": ["discount", "coupon", "$"],
            "max_words": 300,
        },
    },
    {
        "name": "Template Fill: Resume Extraction",
        "pattern": "template_fill",
        "variables": {
            "text": (
                "John Smith is a software engineer at Google with 5 years of experience. "
                "He graduated from MIT with a BS in Computer Science in 2019. "
                "He specializes in distributed systems and Go programming."
            ),
            "template_structure": (
                "Name: [full name]\n"
                "Company: [current employer]\n"
                "Years of Experience: [number]\n"
                "Education: [degree, school, year]\n"
                "Specialties: [comma-separated list]"
            ),
        },
        "criteria": {
            "required_keywords": ["John Smith", "Google", "MIT"],
        },
    },
    {
        "name": "Guardrail: Scoped Assistant",
        "pattern": "guardrail",
        "variables": {
            "role": "Python programming tutor",
            "domain": "Python programming",
            "additional_rules": "Do not write complete solutions. Guide the student with hints.",
            "question": "How do I sort a list of dictionaries by a specific key?",
        },
        "criteria": {
            "required_keywords": ["sorted", "key", "lambda"],
            "forbidden_phrases": ["here is the complete solution"],
        },
    },
]


def run_test_suite():
    print("=" * 70)
    print("  PROMPT ENGINEERING TEST SUITE")
    print("=" * 70)

    all_results = []

    for test in TEST_SUITE:
        print(f"\n{'=' * 60}")
        print(f"  Test: {test['name']}")
        print(f"  Pattern: {test['pattern']}")
        print(f"{'=' * 60}")

        prompt = build_prompt(test["pattern"], test["variables"])
        # print(f"\n  System: {prompt['system'][:80]}...")
        # print(f"  User prompt: {prompt['user'][:120]}...")
        # print(f"  Temperature: {prompt['temperature']}")

        print("Running prompt...")
        results = run_prompt(prompt)
        print("Prompt run completed.")

        print("Comparing model results...")
        comparison, ranked = compare_models(results, test["criteria"])
        print("Comparison completed.")
        
        print(
            f"\n  {'Provider':<12} {'Model':<22} {'Score':>7} {'Words':>7} "
            f"{'Format':>8} {'Keywords':>9} {'Latency':>11}"
        )
        print(f"  {'-' * 82}")
        for provider, data in ranked:
            score = data["scores"]["composite_score"]
            words = data["scores"]["response_words"]
            format_compliant = data["scores"]["format_compliant"]
            format_display = "N/A" if format_compliant is None else ("pass" if format_compliant else "fail")
            keyword_coverage = data["scores"]["keyword_coverage"]
            keyword_display = "N/A" if keyword_coverage is None else f"{keyword_coverage:.0%}"
            latency = data["latency_ms"]
            print(
                f"  {provider:<12} {data['model']:<22} {score:>7.3f} {words:>7} "
                f"{format_display:>8} {keyword_display:>9} {latency:>9.1f}ms"
            )

        all_results.append({
            "test": test["name"],
            "pattern": test["pattern"],
            "models": {
                provider: {
                    "model": data["model"],
                    "score": data["scores"]["composite_score"],
                    "response_chars": data["scores"]["response_chars"],
                    "response_words": data["scores"]["response_words"],
                    "format_compliant": data["scores"]["format_compliant"],
                    "keyword_coverage": data["scores"]["keyword_coverage"],
                    "latency_ms": data["latency_ms"],
                }
                for provider, data in comparison.items()
            },
        })

    print(f"\n\n{'=' * 70}")
    print("  SUMMARY: INSTRUCTION FOLLOWING ACROSS ALL TESTS")
    print(f"{'=' * 70}")

    provider_totals = {}
    for result in all_results:
        for provider, metrics in result["models"].items():
            totals = provider_totals.setdefault(
                provider,
                {
                    "model": metrics["model"],
                    "scores": [],
                    "latencies": [],
                    "formats": [],
                    "keyword_coverages": [],
                },
            )
            totals["scores"].append(metrics["score"])
            totals["latencies"].append(metrics["latency_ms"])
            if metrics["format_compliant"] is not None:
                totals["formats"].append(metrics["format_compliant"])
            if metrics["keyword_coverage"] is not None:
                totals["keyword_coverages"].append(metrics["keyword_coverage"])

    summary = []
    for provider, totals in provider_totals.items():
        average_score = sum(totals["scores"]) / len(totals["scores"])
        average_latency = sum(totals["latencies"]) / len(totals["latencies"])
        format_rate = (
            sum(totals["formats"]) / len(totals["formats"])
            if totals["formats"]
            else None
        )
        keyword_average = (
            sum(totals["keyword_coverages"]) / len(totals["keyword_coverages"])
            if totals["keyword_coverages"]
            else None
        )
        summary.append((provider, totals["model"], average_score))
        format_display = "N/A" if format_rate is None else f"{format_rate:.0%}"
        keyword_display = "N/A" if keyword_average is None else f"{keyword_average:.0%}"
        print(
            f"  {provider:<12} {totals['model']:<22} score={average_score:.3f} "
            f"format={format_display} keywords={keyword_display} "
            f"latency={average_latency:.1f}ms"
        )

    best_score = max(item[2] for item in summary)
    leaders = [item for item in summary if abs(item[2] - best_score) < 1e-9]
    if len(leaders) == 1:
        provider, model, score = leaders[0]
        print(f"\n  Most precise: {model} ({provider}) with an average score of {score:.3f}.")
    else:
        models = ", ".join(f"{model} ({provider})" for provider, model, _ in leaders)
        print(f"\n  Most precise: tie between {models} at {best_score:.3f}.")

    return all_results



if __name__ == "__main__":
    run_test_suite()
