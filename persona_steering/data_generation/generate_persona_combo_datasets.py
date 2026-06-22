import argparse
import json
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

try:
    from ..config import setup_credentials
except ImportError:
    from config import setup_credentials


BASE_DIR = Path(__file__).resolve().parent


ALLOWED_INSTRUCTION_MODES = {"simple", "fewshot"}


def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _flip_instruction_pairs(instruction_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    flipped = []
    for pair in instruction_pairs:
        flipped.append({"pos": pair["neg"], "neg": pair["pos"]})
    return flipped


def _openai_instruction_pairs(client: OpenAI, model: str, prompt: str) -> List[Dict[str, str]]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "instruction_pairs",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 5,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "pos": {"type": "string"},
                                    "neg": {"type": "string"},
                                },
                                "required": ["pos", "neg"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
            },
        },
        messages=[
            {
                "role": "system",
                "content": "Return valid JSON only. No markdown. No extra keys.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    parsed = json.loads(content)
    return parsed["instruction"]


def _build_simple_user_instruction_pairs(user_persona_positive: str, user_persona_negative: str) -> List[Dict[str, str]]:
    return [
        {
            "pos": f"The user persona is {user_persona_positive}. Interpret the user prompts as reflecting that persona consistently.",
            "neg": f"The user persona is {user_persona_negative}. Interpret the user prompts as reflecting that persona consistently.",
        },
        {
            "pos": f"Assume the user has a stable {user_persona_positive} disposition across the full conversation.",
            "neg": f"Assume the user has a stable {user_persona_negative} disposition across the full conversation.",
        },
        {
            "pos": f"Read each prompt as coming from a {user_persona_positive} user, including wording choices and intent.",
            "neg": f"Read each prompt as coming from a {user_persona_negative} user, including wording choices and intent.",
        },
        {
            "pos": f"Anchor the conversation context on a {user_persona_positive} user identity.",
            "neg": f"Anchor the conversation context on a {user_persona_negative} user identity.",
        },
        {
            "pos": f"The user should be represented as {user_persona_positive} in tone, intent, and framing.",
            "neg": f"The user should be represented as {user_persona_negative} in tone, intent, and framing.",
        },
    ]


def _build_simple_joint_instruction_pairs(
    ai_persona_positive: str,
    ai_persona_negative: str,
    user_persona_positive: str,
    user_persona_negative: str,
) -> List[Dict[str, str]]:
    return [
        {
            "pos": (
                f"Assistant persona is {ai_persona_positive}. User persona is {user_persona_positive}. "
                f"Keep the assistant consistently {ai_persona_positive}."
            ),
            "neg": (
                f"Assistant persona is {ai_persona_negative}. User persona is {user_persona_negative}. "
                f"Keep the assistant consistently {ai_persona_negative}."
            ),
        },
        {
            "pos": (
                f"Maintain a stable {ai_persona_positive} assistant style in all turns while the user is {user_persona_positive}."
            ),
            "neg": (
                f"Maintain a stable {ai_persona_negative} assistant style in all turns while the user is {user_persona_negative}."
            ),
        },
        {
            "pos": (
                f"Hold assistant behavior fixed at {ai_persona_positive}; vary only user context to {user_persona_positive}."
            ),
            "neg": (
                f"Hold assistant behavior fixed at {ai_persona_negative}; vary only user context to {user_persona_negative}."
            ),
        },
        {
            "pos": (
                f"Assistant remains {ai_persona_positive} regardless of topic. User is modeled as {user_persona_positive}."
            ),
            "neg": (
                f"Assistant remains {ai_persona_negative} regardless of topic. User is modeled as {user_persona_negative}."
            ),
        },
        {
            "pos": (
                f"Assistant role is permanently {ai_persona_positive}. User-side framing is {user_persona_positive}."
            ),
            "neg": (
                f"Assistant role is permanently {ai_persona_negative}. User-side framing is {user_persona_negative}."
            ),
        },
    ]


def _build_user_fewshot_prompt(
    user_persona_positive: str,
    user_persona_negative: str,
    fewshot_turns: int,
    fewshot_min_words: int,
) -> str:
    return f"""
Create exactly 5 instruction pairs for a USER-persona induction dataset.

Goal:
- POS means the user is {user_persona_positive}.
- NEG means the user is {user_persona_negative}.
- Keep assistant behavior neutral/generic in BOTH POS and NEG.

Critical formatting constraints for EVERY pos and neg instruction string:
1) Start with a short 'Persona summary' sentence.
2) Include a 'Behavior rules:' section with at least 3 bullet-style rule lines.
3) Include a 'Few-shot dialogue:' section with at least {fewshot_turns} turns, alternating User: and Assistant:.
4) Each instruction must be at least {fewshot_min_words} words.
5) The examples must be realistic, concrete, and varied across the 5 pairs.
6) Do not use markdown code fences.

Output JSON schema:
{{"instruction": [{{"pos":"...","neg":"..."}} x 5]}}
""".strip()


def _build_joint_fewshot_prompt(
    ai_persona_positive: str,
    ai_persona_negative: str,
    user_persona_positive: str,
    user_persona_negative: str,
    fewshot_turns: int,
    fewshot_min_words: int,
) -> str:
    return f"""
Create exactly 5 instruction pairs for a JOINT persona induction dataset.

Goal:
- POS: assistant is {ai_persona_positive}, user is {user_persona_positive}.
- NEG: assistant is {ai_persona_negative}, user is {user_persona_negative}.
- IMPORTANT: assistant persona must match POS/NEG setting in each side.

Critical formatting constraints for EVERY pos and neg instruction string:
1) Start with a short 'Assistant+User summary' sentence.
2) Include a 'Behavior rules:' section with at least 3 bullet-style rule lines.
3) Include a 'Few-shot dialogue:' section with at least {fewshot_turns} turns, alternating User: and Assistant:.
4) Each instruction must be at least {fewshot_min_words} words.
5) In the dialogue examples, assistant style must be {ai_persona_positive} for POS and {ai_persona_negative} for NEG.
6) Do not use markdown code fences.

Output JSON schema:
{{"instruction": [{{"pos":"...","neg":"..."}} x 5]}}
""".strip()


def generate_persona_combo_datasets(
    model: str,
    source_ai_dataset: str,
    source_eval_dataset: str,
    ai_output_name: str,
    user_output_name: str,
    joint_output_name: str,
    ai_persona_positive: str,
    ai_persona_negative: str,
    user_persona_positive: str,
    user_persona_negative: str,
    user_instruction_mode: str = "fewshot",
    joint_instruction_mode: str = "fewshot",
    fewshot_turns: int = 6,
    fewshot_min_words: int = 140,
) -> Dict[str, str]:
    """
    Generate three extraction datasets:
    1) AI dataset by flipping source AI dataset instruction pairs.
    2) User dataset in either simple mode or extensive few-shot mode.
    3) Joint dataset in either simple mode or extensive few-shot mode.

    Questions/eval prompt are inherited from source_eval_dataset to stay pipeline-compatible.
    """
    if user_instruction_mode not in ALLOWED_INSTRUCTION_MODES:
        raise ValueError(f"user_instruction_mode must be one of {sorted(ALLOWED_INSTRUCTION_MODES)}")
    if joint_instruction_mode not in ALLOWED_INSTRUCTION_MODES:
        raise ValueError(f"joint_instruction_mode must be one of {sorted(ALLOWED_INSTRUCTION_MODES)}")

    if fewshot_turns < 4:
        raise ValueError("fewshot_turns must be >= 4")
    if fewshot_min_words < 80:
        raise ValueError("fewshot_min_words must be >= 80")

    needs_openai = user_instruction_mode == "fewshot" or joint_instruction_mode == "fewshot"
    client = None
    if needs_openai:
        cfg = setup_credentials()
        client = OpenAI(api_key=cfg.openai_api_key)

    extract_dir = BASE_DIR / "trait_data_extract"
    eval_dir = BASE_DIR / "trait_data_eval"

    ai_src = _load_json(extract_dir / source_ai_dataset)
    eval_src = _load_json(eval_dir / source_eval_dataset)

    ai_data = {
        "instruction": _flip_instruction_pairs(ai_src["instruction"]),
        "questions": eval_src["questions"],
        "eval_prompt": eval_src["eval_prompt"],
    }

    if user_instruction_mode == "simple":
        user_instructions = _build_simple_user_instruction_pairs(
            user_persona_positive=user_persona_positive,
            user_persona_negative=user_persona_negative,
        )
    else:
        user_prompt = _build_user_fewshot_prompt(
            user_persona_positive=user_persona_positive,
            user_persona_negative=user_persona_negative,
            fewshot_turns=fewshot_turns,
            fewshot_min_words=fewshot_min_words,
        )
        user_instructions = _openai_instruction_pairs(client, model, user_prompt)

    if joint_instruction_mode == "simple":
        joint_instructions = _build_simple_joint_instruction_pairs(
            ai_persona_positive=ai_persona_positive,
            ai_persona_negative=ai_persona_negative,
            user_persona_positive=user_persona_positive,
            user_persona_negative=user_persona_negative,
        )
    else:
        joint_prompt = _build_joint_fewshot_prompt(
            ai_persona_positive=ai_persona_positive,
            ai_persona_negative=ai_persona_negative,
            user_persona_positive=user_persona_positive,
            user_persona_negative=user_persona_negative,
            fewshot_turns=fewshot_turns,
            fewshot_min_words=fewshot_min_words,
        )
        joint_instructions = _openai_instruction_pairs(client, model, joint_prompt)

    user_data = {
        "instruction": user_instructions,
        "questions": eval_src["questions"],
        "eval_prompt": eval_src["eval_prompt"],
    }
    joint_data = {
        "instruction": joint_instructions,
        "questions": eval_src["questions"],
        "eval_prompt": eval_src["eval_prompt"],
    }

    ai_path = extract_dir / f"{ai_output_name}.json"
    user_path = extract_dir / f"{user_output_name}.json"
    joint_path = extract_dir / f"{joint_output_name}.json"

    _save_json(ai_path, ai_data)
    _save_json(user_path, user_data)
    _save_json(joint_path, joint_data)

    # Also place joint in eval folder so evaluate can run directly with same trait key.
    _save_json(eval_dir / f"{joint_output_name}.json", joint_data)

    return {
        "ai_dataset": str(ai_path),
        "user_dataset": str(user_path),
        "joint_dataset_extract": str(joint_path),
        "joint_dataset_eval": str(eval_dir / f"{joint_output_name}.json"),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate persona-combination datasets for extract/eval pipeline")
    parser.add_argument("--model", type=str, default="gpt-5.4-nano")
    parser.add_argument("--source_ai_dataset", type=str, default="orig_impolite.json")
    parser.add_argument("--source_eval_dataset", type=str, default="impolite.json")

    parser.add_argument("--ai_output_name", type=str, default="orig_polite")
    parser.add_argument("--user_output_name", type=str, default="user_impolite_fewshot")
    parser.add_argument("--joint_output_name", type=str, default="polite_ai_user_impolite_fewshot")

    parser.add_argument("--ai_persona_positive", type=str, default="polite")
    parser.add_argument("--ai_persona_negative", type=str, default="impolite")
    parser.add_argument("--user_persona_positive", type=str, default="impolite")
    parser.add_argument("--user_persona_negative", type=str, default="polite")
    parser.add_argument("--user_instruction_mode", type=str, choices=sorted(ALLOWED_INSTRUCTION_MODES), default="fewshot")
    parser.add_argument("--joint_instruction_mode", type=str, choices=sorted(ALLOWED_INSTRUCTION_MODES), default="fewshot")
    parser.add_argument("--fewshot_turns", type=int, default=6)
    parser.add_argument("--fewshot_min_words", type=int, default=140)

    args = parser.parse_args()

    paths = generate_persona_combo_datasets(
        model=args.model,
        source_ai_dataset=args.source_ai_dataset,
        source_eval_dataset=args.source_eval_dataset,
        ai_output_name=args.ai_output_name,
        user_output_name=args.user_output_name,
        joint_output_name=args.joint_output_name,
        ai_persona_positive=args.ai_persona_positive,
        ai_persona_negative=args.ai_persona_negative,
        user_persona_positive=args.user_persona_positive,
        user_persona_negative=args.user_persona_negative,
        user_instruction_mode=args.user_instruction_mode,
        joint_instruction_mode=args.joint_instruction_mode,
        fewshot_turns=args.fewshot_turns,
        fewshot_min_words=args.fewshot_min_words,
    )

    print("Generated datasets:")
    for k, v in paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
