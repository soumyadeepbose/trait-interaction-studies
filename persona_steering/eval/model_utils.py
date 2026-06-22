import os, re, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import socket





def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  
        return s.getsockname()[1]


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# ---------- 工具 ----------
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")

def _pick_latest_checkpoint(model_path: str) -> str:
    ckpts = [(int(m.group(1)), p) for p in Path(model_path).iterdir()
             if (m := _CHECKPOINT_RE.fullmatch(p.name)) and p.is_dir()]
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else model_path

def _is_lora(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()

def _load_and_merge_lora(lora_path: str, dtype, device_map):
    cfg = PeftConfig.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, torch_dtype=dtype, device_map=device_map
    )
    return PeftModel.from_pretrained(base, lora_path).merge_and_unload()

def _load_tokenizer(path_or_id: str):
    tok = AutoTokenizer.from_pretrained(path_or_id)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok

def load_model(model_path: str, dtype=torch.bfloat16):
    if not os.path.exists(model_path):               # ---- Hub ----
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(model_path)
        return model, tok

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        model = _load_and_merge_lora(resolved, dtype, "auto")
        tok = _load_tokenizer(model.config._name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(resolved)
    return model, tok

def load_vllm_model(model_path: str):
    from vllm import LLM

    gpu_mem_util = _env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.9)
    max_num_seqs = _env_int("VLLM_MAX_NUM_SEQS", 32)
    max_model_len_hub = _env_int("VLLM_MAX_MODEL_LEN_HUB", 1000)
    max_model_len_local = _env_int("VLLM_MAX_MODEL_LEN_LOCAL", 20000)

    if not os.path.exists(model_path):               # ---- Hub ----
        llm = LLM(
            model=model_path,
            enable_prefix_caching=True,
            enable_lora=True,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len_hub,
            max_lora_rank=128,
            trust_remote_code=True, 
            enforce_eager=True
        )
        tok = llm.get_tokenizer()
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        return llm, tok, None

    # ---- 本地 ----
    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    is_lora = _is_lora(resolved)

    base_path = (PeftConfig.from_pretrained(resolved).base_model_name_or_path
                 if is_lora else resolved)

    llm = LLM(
        model=base_path,
        enable_prefix_caching=True,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len_local,
        max_lora_rank=128,
    )

    if is_lora:
        lora_path = resolved
    else:
        lora_path = None

    tok = llm.get_tokenizer()
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return llm, tok, lora_path
