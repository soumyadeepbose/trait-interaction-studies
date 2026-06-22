gpu=${1:-0}


CUDA_VISIBLE_DEVICES=$gpu python -m persona_steering.eval.cal_projection \
    --file_path persona_steering/eval_persona_eval/Qwen2.5-7B-Instruct/evil.csv \
    --vector_path persona_steering/persona_vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt \
    --layer 20 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --projection_type proj