#!/usr/bin/env bash
#
# run.sh — end-to-end persona-interaction pipeline (factorial + refusal + tensor).
#
# Usage:
#   ./run.sh                 # full single-pair pipeline (build → … → behavioral law)
#   ./run.sh <stage>         # run one stage: build extract combine refusal metrics cosines law tensor
#   ./run.sh all             # everything including tensor (needs the full trait grid)
#
# Override any config var inline, e.g.:
#   MODEL=Qwen/Qwen2.5-7B SHORT=Qwen2.5-7B ./run.sh          # base model
#   SETTING=ai_ai PREFIX=fac_humor_evil_ai ./run.sh          # AI+AI control
#
set -euo pipefail

# ── Config (edit or override via env) ─────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SHORT="${SHORT:-Qwen2.5-7B-Instruct}"
SETTING="${SETTING:-ai_user}"               # ai_user | ai_ai
PREFIX="${PREFIX:-fac_humor_evil_user}"     # cell prefix; cells are ${PREFIX}_{pp,pn,np,nn}
LIMIT_Q="${LIMIT_Q:-64}"                    # questions per cell
MAX_TOK="${MAX_TOK:-128}"
COEFFS="${COEFFS:-0,5,10,20}"               # behavioral-law steering sweep (large; unit-norm steering)
ABLATE_LAYERS="${ABLATE_LAYERS:-15-28}"
# Tensor grid (only used by the `tensor` stage; extract every pair first as fac_<ai>_<user>):
AI_TRAITS="${AI_TRAITS:-humour,evil,formal}"
USER_TRAITS="${USER_TRAITS:-evil,polite,anxious}"
BASE_TENSOR="${BASE_TENSOR:-output/tensor_Qwen2.5-7B.pt}"   # for the delta stage

# ── Derived paths ─────────────────────────────────────────────────────────────
DATA=persona_steering/data_generation/trait_data_extract
REFUSAL_DIR=persona_steering/data_generation/refusal
VEC=persona_steering/persona_vectors/$SHORT
EXTRACT=persona_steering/eval_persona_extract/$SHORT
OUT=output/metrics_factorial
TENSOR_DIR=output/tensor_$SHORT
LAW_DIR=output/ablation_law

log(){ printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die(){ printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "${OPENAI_API_KEY:-}" ] || echo "WARNING: OPENAI_API_KEY is not set; the judge calls will fail."
mkdir -p "$VEC" "$EXTRACT" "$OUT" "$TENSOR_DIR" "$LAW_DIR" "$REFUSAL_DIR"

# ── Stages ────────────────────────────────────────────────────────────────────

stage_build(){
  log "1. Build datasets ($SETTING, prefix=$PREFIX)"
  python build_factorial_dataset.py --setting "$SETTING" --out_dir "$DATA" --prefix "$PREFIX"
  [ -f "$REFUSAL_DIR/refusal.json" ] || \
    python build_refusal_dataset.py --out_dir "$REFUSAL_DIR" --n 128
}

stage_extract(){
  log "2. Extract 4 factorial cell vectors (threshold 0, n_per_question 1)"
  for CELL in pp pn np nn; do
    echo "--- cell $CELL ---"
    python -m persona_steering.eval.eval_persona --model "$MODEL" --trait "${PREFIX}_${CELL}" \
      --output_path "$EXTRACT/${PREFIX}_${CELL}_pos_instruct.csv" \
      --persona_instruction_type pos --assistant_name "${PREFIX}_${CELL}" \
      --version extract --n_per_question 1 --max_tokens "$MAX_TOK" --limit_questions "$LIMIT_Q" --overwrite

    python -m persona_steering.eval.eval_persona --model "$MODEL" --trait "${PREFIX}_${CELL}" \
      --output_path "$EXTRACT/${PREFIX}_${CELL}_neg_instruct.csv" \
      --persona_instruction_type neg --assistant_name "${PREFIX}_${CELL}" \
      --version extract --n_per_question 1 --max_tokens "$MAX_TOK" --limit_questions "$LIMIT_Q" --overwrite

    python persona_steering/generate_vec.py --model_name "$MODEL" \
      --pos_path "$EXTRACT/${PREFIX}_${CELL}_pos_instruct.csv" \
      --neg_path "$EXTRACT/${PREFIX}_${CELL}_neg_instruct.csv" \
      --trait "${PREFIX}_${CELL}" --save_dir "$VEC" \
      --threshold 0 --coherence_threshold 0
  done
}

stage_combine(){
  log "3. Combine cells → tau_A, tau_U, joint(R)"
  python factorial_residue.py --vec_dir "$VEC" --prefix "$PREFIX"
}

stage_refusal(){
  log "4. Extract refusal direction"
  python extract_refusal_direction.py --model_name "$MODEL" \
    --refusal_json "$REFUSAL_DIR/refusal.json" \
    --save_dir "$VEC" --aggregate last_token
}

stage_metrics(){
  log "5. Per-layer metrics"
  python metrics.py --vec_dir "$VEC" \
    --ai_trait "${PREFIX}_fac_tauA" --user_trait "${PREFIX}_fac_tauU" --joint_trait "${PREFIX}_fac_joint" \
    --output_dir "$OUT" --output_prefix "$PREFIX"
}

ensure_refusal_cosines(){
  # Materialize the cosine-battery helper if it isn't committed yet.
  [ -f refusal_cosines.py ] && return
  cat > refusal_cosines.py <<'PY'
import sys, torch, pandas as pd, torch.nn.functional as F
from pathlib import Path
vec_dir, fac, out_csv = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
L = lambda n: torch.load(vec_dir / f"{n}_response_avg_diff.pt", map_location="cpu", weights_only=False).float()
tau_A, tau_U, joint, d_ref = L(f"{fac}_tauA"), L(f"{fac}_tauU"), L(f"{fac}_joint"), L("refusal")
R = joint - tau_A - tau_U
cos = lambda a, b: F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
def gs(v1, v2, eps=1e-8):
    e1 = v1 / (v1.norm() + eps); v2o = v2 - (v2 @ e1) * e1; n2 = v2o.norm()
    return e1, (v2o / (n2 + eps) if n2 > eps else torch.zeros_like(e1))
rows = []
for l in range(min(R.shape[0], d_ref.shape[0])):
    e1, e2 = gs(tau_A[l], tau_U[l])
    R_perp = R[l] - (R[l] @ e1) * e1 - (R[l] @ e2) * e2
    rows.append(dict(layer=l, cos_R_refusal=cos(R[l], d_ref[l]),
                     cos_Rperp_refusal=cos(R_perp, d_ref[l]),
                     cos_tauA_refusal=cos(tau_A[l], d_ref[l]),
                     cos_tauU_refusal=cos(tau_U[l], d_ref[l]),
                     cos_joint_refusal=cos(joint[l], d_ref[l])))
pd.DataFrame(rows).to_csv(out_csv, index=False); print("wrote", out_csv)
PY
}

stage_cosines(){
  log "6. Refusal cosine battery"
  ensure_refusal_cosines
  python refusal_cosines.py "$VEC" "${PREFIX}_fac" "$OUT/${PREFIX}_refusal_cosines.csv"
}

stage_law(){
  log "8. Behavioral law (amplify_user sweep → judge → correlate)"
  python ablation_study.py --model_name "$MODEL" --modes amplify_user \
    --user_vec "$VEC/${PREFIX}_fac_joint_response_avg_diff.pt" \
    --coeff "$COEFFS" --intervention_layers "$ABLATE_LAYERS" \
    --trait_json "$DATA/${PREFIX}_pp.json" --persona_side pos \
    --n_per_question 1 --instruction_indices 0 \
    --output_dir "$LAW_DIR" --output_csv generations.csv

  python score_ablation_outputs.py --input_csv "$LAW_DIR/generations.csv" \
    --answer_col answer --overwrite

  python interaction_tensor.py behavioral_law \
    --csv "$LAW_DIR/generations_scored.csv" \
    --residue "$VEC/${PREFIX}_fac_joint_response_avg_diff.pt" \
    --ai_vec  "$VEC/${PREFIX}_fac_tauA_response_avg_diff.pt" \
    --coeff_col coeff --score_col harmfulness --out "$LAW_DIR/law.csv"
}

stage_tensor(){
  log "7. Interaction tensor (needs every fac_<ai>_<user> pair extracted first)"
  python interaction_tensor.py assemble --vec_dir "$VEC" \
    --ai_traits "$AI_TRAITS" --user_traits "$USER_TRAITS" --out "$TENSOR_DIR.pt"
  python interaction_tensor.py decompose --tensor "$TENSOR_DIR.pt" --rank 4 --outdir "$TENSOR_DIR"
  python interaction_tensor.py refusal_map --tensor "$TENSOR_DIR.pt" \
    --refusal "$VEC/refusal_response_avg_diff.pt" --out "$TENSOR_DIR/refusal_map.csv"
  if [ -f "$BASE_TENSOR" ]; then
    python interaction_tensor.py delta --tensor_instruct "$TENSOR_DIR.pt" --tensor_base "$BASE_TENSOR" \
      --refusal "$VEC/refusal_response_avg_diff.pt" --out "$TENSOR_DIR/delta_T.pt"
  else
    echo "  (skipping delta: base tensor $BASE_TENSOR not found)"
  fi
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
CORE=(build extract combine refusal metrics cosines law)

case "${1:-pipeline}" in
  build|extract|combine|refusal|metrics|cosines|law|tensor) "stage_$1" ;;
  pipeline) for s in "${CORE[@]}"; do "stage_$s"; done ;;
  all)      for s in "${CORE[@]}"; do "stage_$s"; done; stage_tensor ;;
  *) die "Unknown stage '$1'. Use: ${CORE[*]} tensor | pipeline | all" ;;
esac

log "Done."