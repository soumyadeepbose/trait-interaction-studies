import argparse
import asyncio
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from persona_steering.eval.prompts import Prompts
    from persona_steering.judge import OpenAiJudge
except ImportError:
    from eval.prompts import Prompts
    from judge import OpenAiJudge


def _sanitize_text(value: object) -> str:
    """Normalize text to avoid malformed JSON payloads in judge API calls."""
    text = str(value) if value is not None else ""
    text = text.replace("\x00", " ")
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return "".join(ch if ch in "\n\t" or ord(ch) >= 32 else " " for ch in text)


# ── Scorer registry: maps friendly names to prompt keys and output column names ──
SCORERS = {
    "coherence": {
        "prompt_key": "coherence_0_100",
        "column_name": "coherence",
        "requires_system": False,
    },
    "adherence": {
        "prompt_key": "adherence_with_system_0_100",
        "column_name": "adherence",
        "requires_system": True,
    },
    "evilness": {
        "prompt_key": "trait_evil_0_100",
        "column_name": "evilness",
        "requires_system": False,
    },
    "harmfulness": {
        "prompt_key": "trait_harmfulness_0_100",
        "column_name": "harmfulness",
        "requires_system": False,
    },
    "manipulativeness": {
        "prompt_key": "trait_manipulativeness_0_100",
        "column_name": "manipulativeness",
        "requires_system": False,
    },
}


async def score_rows(
    df: pd.DataFrame,
    question_col: str,
    answer_col: str,
    system_col: str,
    judge_model: str,
    max_concurrent_judges: int,
    scorer_keys: list[str],  # list of scorer names to compute
):
    """Score rows using selected scorers.
    
    Args:
        df: input dataframe with question, answer, (optionally) system_message columns.
        question_col: name of the question column.
        answer_col: name of the answer column.
        system_col: name of the system message column (may not exist; required for "adherence" scorer).
        judge_model: LLM to use as judge.
        max_concurrent_judges: semaphore limit for concurrent API calls.
        scorer_keys: list of scorer names to compute (e.g., ["coherence", "evilness", "harmfulness"]).
    
    Returns:
        dict[scorer_name] -> list of scores.
    """
    # Validate scorer keys and load prompts
    for scorer_name in scorer_keys:
        if scorer_name not in SCORERS:
            raise ValueError(
                f"Unknown scorer '{scorer_name}'. Available: {list(SCORERS.keys())}"
            )
        prompt_key = SCORERS[scorer_name]["prompt_key"]
        if prompt_key not in Prompts:
            raise KeyError(f"Missing prompt template: {prompt_key}")

    # Create judges (one per scorer)
    judges = {}
    for scorer_name in scorer_keys:
        prompt_key = SCORERS[scorer_name]["prompt_key"]
        judges[scorer_name] = OpenAiJudge(
            judge_model, Prompts[prompt_key], eval_type="0_100"
        )

    semaphore = asyncio.Semaphore(max_concurrent_judges)
    scores = {scorer_name: [None] * len(df) for scorer_name in scorer_keys}

    async def run_one(row_idx: int):
        async with semaphore:
            row = df.iloc[row_idx]
            question = _sanitize_text(row[question_col])
            answer = _sanitize_text(row[answer_col])
            system_message = ""
            if system_col in df.columns and pd.notna(row[system_col]):
                system_message = _sanitize_text(row[system_col])

            row_scores = {}
            for scorer_name, judge in judges.items():
                scorer_info = SCORERS[scorer_name]
                if scorer_info["requires_system"]:
                    score = await judge(
                        question=question, answer=answer, system_message=system_message
                    )
                else:
                    score = await judge(question=question, answer=answer)
                row_scores[scorer_name] = score

            return row_idx, row_scores

    tasks = [run_one(i) for i in range(len(df))]

    with tqdm(total=len(tasks), desc="Scoring rows") as pbar:
        for task in asyncio.as_completed(tasks):
            row_idx, row_scores = await task
            for scorer_name, score in row_scores.items():
                scores[scorer_name][row_idx] = score
            pbar.update(1)

    return scores


def build_output_path(input_csv: Path, output_csv: str | None) -> Path:
    if output_csv:
        return Path(output_csv)
    return input_csv.with_name(f"{input_csv.stem}_scored{input_csv.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Score ablation outputs with flexible scorer selection. "
        "Default: coherence,adherence,evilness,harmfulness (evilness is the NEW trait metric)."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--max_concurrent_judges", type=int, default=5)
    parser.add_argument("--question_col", type=str, default="question")
    parser.add_argument("--answer_col", type=str, default="answer")
    parser.add_argument("--system_col", type=str, default="system_message")
    parser.add_argument(
        "--scorers",
        type=str,
        default="coherence,evilness,adherence",
        help=(
            "Comma-separated list of scorers to compute. "
            f"Available: {', '.join(SCORERS.keys())}. "
            "Default includes evilness (the NEW trait metric), "
            "distinct from harmfulness (practical harm)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv = build_output_path(input_csv, args.output_csv)
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output CSV already exists: {output_csv}. Use --overwrite to replace it."
        )

    # Parse scorer list
    scorer_names = [s.strip() for s in args.scorers.split(",")]
    if not scorer_names:
        raise ValueError("--scorers must specify at least one scorer")

    df = pd.read_csv(input_csv)
    for col in [args.question_col, args.answer_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {input_csv}")

    # Run scoring
    scores_dict = asyncio.run(
        score_rows(
            df=df,
            question_col=args.question_col,
            answer_col=args.answer_col,
            system_col=args.system_col,
            judge_model=args.judge_model,
            max_concurrent_judges=args.max_concurrent_judges,
            scorer_keys=scorer_names,
        )
    )

    # Add score columns to dataframe
    for scorer_name, scores_list in scores_dict.items():
        col_name = SCORERS[scorer_name]["column_name"]
        df[col_name] = scores_list

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved scored CSV: {output_csv}")
    print(f"Rows: {len(df)}")
    for scorer_name, scores_list in scores_dict.items():
        col_name = SCORERS[scorer_name]["column_name"]
        mean_score = pd.Series(scores_list).mean()
        print(f"{col_name:20s} mean: {mean_score:.2f}")


if __name__ == "__main__":
    main()