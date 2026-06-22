gpu=${1:-0}


CUDA_VISIBLE_DEVICES=$gpu python -m persona_steering.eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path persona_steering/eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=$gpu python -m persona_steering.eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path persona_steering/eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=$gpu python persona_steering/generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path persona_steering/eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --neg_path persona_steering/eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_steering/persona_vectors/Qwen2.5-7B-Instruct/ \
    --threshold 50