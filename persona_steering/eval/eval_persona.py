import os
import asyncio
import yaml
from typing import Dict, List
import json
from pathlib import Path
import torch
import pandas as pd
import random
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams

try:
    from ..judge import OpenAiJudge
    from ..activation_steer import ActivationSteerer
    from .model_utils import load_model, load_vllm_model
    from .prompts import Prompts
    from ..config import setup_credentials
except ImportError:
    from judge import OpenAiJudge
    from activation_steer import ActivationSteerer
    from eval.model_utils import load_model, load_vllm_model
    from eval.prompts import Prompts
    from config import setup_credentials

from itertools import islice
import asyncio
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()

PERSONA_ROOT = Path(__file__).resolve().parents[1]


def _resolve_trait_data_path(version: str, trait: str) -> Path:
    path = PERSONA_ROOT / "data_generation" / f"trait_data_{version}" / f"{trait}.json"
    if path.exists():
        return path
    legacy = Path(f"data_generation/trait_data_{version}/{trait}.json")
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"Trait data not found for trait='{trait}', version='{version}'. "
        f"Checked {path} and {legacy}."
    )

def sample_steering(model, tokenizer, conversations,  vector, layer, coef, bs=20, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, steering_type="response"):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for messages in conversations:
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    
    outputs = []
    for i in trange(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer-1, positions=steering_type):
            with torch.no_grad():
                output = model.generate(**tokenized_batch, do_sample=(temperature > 0), temperature=temperature, top_p=top_p, max_new_tokens=max_tokens,use_cache=True, min_new_tokens=min_tokens)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)
    return prompts, outputs


def sample(model, tokenizer, conversations, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, lora_path=None):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=min_tokens
    )

    texts = []
    for i, messages in enumerate(conversations):
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    if lora_path:
        completions = model.generate(texts, **generate_kwargs, lora_request=LoRARequest("default", 1, lora_path=lora_path))
    else:
        completions = model.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return texts, answers



def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o",
            judge_eval_type: str = "0_100",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt, eval_type=judge_eval_type if metric != "coherence" else "0_100") for metric, prompt in judge_prompts.items()}
    
    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role='user', content=i)] for i in paraphrases]
        if self.system:
            conversations = [[dict(role='system', content=self.system)] + c for c in conversations]
        return paraphrases, conversations
    
    async def eval(self, llm, tokenizer, coef, vector=None, layer=None, max_tokens=1000, n_per_question=100, steering_type="last", lora_path=None):
        paraphrases, conversations = self.get_input(n_per_question)
        if coef != 0:
            prompts, answers = sample_steering(llm, tokenizer, conversations, vector, layer, coef, temperature=self.temperature, max_tokens=max_tokens, steering_type=steering_type)
        else:
            prompts, answers = sample(llm, tokenizer, conversations, temperature=self.temperature, max_tokens=max_tokens, lora_path=lora_path)
        df = pd.DataFrame([
            dict(question=question,prompt=prompt, answer=answer, question_id=self.id)
            for question, answer, prompt in zip(paraphrases, answers, prompts)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(paraphrases, answers)
            ])
            df[score] = scores
        return df
        

def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def _parse_score_traits(score_traits) -> list[str]:
    if score_traits is None:
        return []
    if isinstance(score_traits, str):
        return [x.strip() for x in score_traits.split(",") if x.strip()]
    if isinstance(score_traits, (list, tuple, set)):
        out = []
        for x in score_traits:
            if x is None:
                continue
            xs = str(x).strip()
            if xs:
                out.append(xs)
        return out
    return []


def _build_extra_judge_prompts(eval_type: str, score_traits=None, include_future_scorers: bool = False) -> dict[str, str]:
    prompts = {}
    parsed_traits = _parse_score_traits(score_traits)
    for trait_name in parsed_traits:
        key = f"trait_{trait_name}_{eval_type}"
        if key not in Prompts:
            raise KeyError(
                f"Missing manual prompt '{key}' in eval/prompts.py. "
                f"Please add it before using --score_traits {trait_name}."
            )
        prompts[f"score_{trait_name}"] = Prompts[key]

    if include_future_scorers:
        for metric in ["misalignment", "adherence"]:
            key = f"{metric}_{eval_type}"
            if key not in Prompts:
                raise KeyError(
                    f"Missing prompt '{key}' in eval/prompts.py required for future scorers."
                )
            prompts[metric] = Prompts[key]

    return prompts


def load_persona_questions(
    trait,
    temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    eval_type="0_100",
    version="eval",
    steer="ai",
    max_tokens=128,
    limit_questions=0,
    score_traits=None,
    include_future_scorers: bool = False,
):
    trait_path = _resolve_trait_data_path(version, trait)
    trait_data = json.load(open(trait_path, "r"))
    judge_prompts = {}
    prompt_template = trait_data["eval_prompt"]
    judge_prompts[trait] = prompt_template
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]
    judge_prompts.update(
        _build_extra_judge_prompts(
            eval_type=eval_type,
            score_traits=score_traits,
            include_future_scorers=include_future_scorers,
        )
    )
    raw_questions = trait_data["questions"]
    if int(limit_questions) > 0:
        raw_questions = raw_questions[: int(limit_questions)]
    questions = []
    for i, question in enumerate(raw_questions):
        if persona_instructions_type is not None:
            persona_instructions = [x[persona_instructions_type] for x in trait_data["instruction"]]
            for k, instruction in enumerate(persona_instructions):
                token_budget = max(1, int(max_tokens) - 13)
                system = (
                    f"{instruction} "
                    f"Keep your response within {token_budget} tokens. "
                    "Do not exceed this limit."
                )
                print("########### The system message is, \n\t", system)
                questions.append(Question(paraphrases=[question], id=f"{trait}_{i}_{persona_instructions_type}_{k}", judge_prompts=judge_prompts, judge=judge_model, temperature=temperature, system=system, judge_eval_type=eval_type ))
        else:
            questions.append(Question(paraphrases=[question], id=f"{trait}_{i}", judge_prompts=judge_prompts, judge=judge_model, temperature=temperature, judge_eval_type=eval_type ))
    return questions





async def eval_batched(questions, llm, tokenizer, coef, vector=None, layer=None, n_per_question=100, max_concurrent_judges=100, max_tokens=1000, steering_type="last", lora_path=None):
    """Batch process all questions together for faster inference"""
    # Collect all prompts from all questions
    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))
    
    # Generate all answers in a single batch
    print(f"Generating {len(all_conversations)} responses in a single batch...")
    if coef != 0:
        prompts, answers = sample_steering(llm, tokenizer, all_conversations, vector, layer, coef, temperature=questions[0].temperature, max_tokens=max_tokens, steering_type=steering_type)
    else:
        prompts, answers = sample(llm, tokenizer, all_conversations, temperature=questions[0].temperature, max_tokens=max_tokens, lora_path=lora_path)
    
    # Prepare data structures for batch evaluation
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []  # Store (question_idx, metric, sample_idx) for each task
    
    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        # Get this question's data
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]
        
        # Create dataframe for this question
        df = pd.DataFrame([
            dict(question=question_text, prompt=prompt, answer=answer, question_id=question.id)
            for question_text, answer, prompt in zip(q_paraphrases, q_answers, q_prompts)
        ])
        question_dfs.append(df)
        
        # Collect all judge tasks
        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(zip(q_paraphrases, q_answers)):
                all_judge_tasks.append((judge, question_text, answer))
                all_judge_indices.append((i, metric, sample_idx))
    
    # Run judge evaluations with concurrency control
    print(f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests...")
    all_results = [None] * len(all_judge_tasks)  # Pre-allocate results array
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent_judges)
    
    async def run_with_semaphore(task_idx, judge, question_text, answer):
        async with semaphore:
            result = await judge(question=question_text, answer=answer)
            return task_idx, result
    
    # Create all tasks with semaphore control
    tasks = [run_with_semaphore(task_idx, judge, question_text, answer) 
             for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)]
    
    # Process tasks in batches with progress bar
    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result  # Store result in correct position
            pbar.update(1)
    
    # Distribute results back to the appropriate dataframes
    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result
    
    return question_dfs

def main(
    model,
    trait,
    output_path,
    coef=0,
    vector_path=None,
    layer=None,
    steering_type="response",
    max_tokens=128,
    n_per_question=10,
    limit_questions=0,
    batch_process=True,
    max_concurrent_judges=10,
    persona_instruction_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-nano",
    version="extract",
    overwrite=False,
    steer_who="ai",
    score_traits=None,
    include_future_scorers=False,
):
    """Evaluate a model on all questions form the evaluation yaml file"""
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        base_metrics = [trait, "coherence"]
        extra_metrics = [c for c in df.columns if c.startswith("score_") or c in ["misalignment", "adherence"]]
        for metric in base_metrics + extra_metrics:
            if metric in df.columns:
                print(f"{metric}:  {df[metric].mean():.2f} +- {df[metric].std():.2f}")
        return
    
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if n_per_question == 1:
        temperature = 0.0
    else:
        temperature = 1.0
    if coef != 0:
        llm, tokenizer = load_model(model)
        lora_path = None
        print("############## coeff is not 0")
        vector = torch.load(vector_path, weights_only=False)[layer]
            
    else:
        print("############## Before model loading")
        llm, tokenizer, lora_path = load_vllm_model(model)
        print("############## Model loaded")
        vector=None
    questions = load_persona_questions(
        trait,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version=version,
        steer=steer_who,
        max_tokens=max_tokens,
        limit_questions=limit_questions,
        score_traits=score_traits,
        include_future_scorers=include_future_scorers,
    )
    if batch_process:
        print(f"Batch processing {len(questions)} '{trait}' questions...")
        outputs_list = asyncio.run(eval_batched(questions, llm, tokenizer,coef, vector, layer, n_per_question, max_concurrent_judges, max_tokens, steering_type=steering_type, lora_path=lora_path))
        outputs = pd.concat(outputs_list)
    else:
        outputs = []
        for question in tqdm(questions, desc=f"Processing {trait} questions"):
            outputs.append(asyncio.run(question.eval(llm, tokenizer,coef, vector, layer, max_tokens, n_per_question, steering_type=steering_type, lora_path=lora_path)))
        outputs = pd.concat(outputs)
    outputs.to_csv(output_path, index=False)
    print(output_path)
    base_metrics = [trait, "coherence"]
    extra_metrics = [c for c in outputs.columns if c.startswith("score_") or c in ["misalignment", "adherence"]]
    for metric in base_metrics + extra_metrics:
        if metric in outputs.columns:
            print(f"{metric}:  {outputs[metric].mean():.2f} +- {outputs[metric].std():.2f}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
