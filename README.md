# trait-interaction-studies

Mechanistic interpretability of AI×user **trait interactions** in LLM activation space:
factorial interaction residue `R`, refusal-direction alignment, and (attempted to) the second-order
**interaction tensor** `T`.

## Setup

```bash
git clone https://github.com/soumyadeepbose/trait-interaction-studies.git
cd trait-interaction-studies
uv sync --link-mode=copy
export OPENAI_API_KEY=sk-...      # used by the judge in eval_persona / score_ablation_outputs
```

## Quick Method

If you want to grab some popcorn and watch the whole thing run autonomously, run:
```bash
chmod +x run.sh
./run.sh 
```
If any argument needs modificatio, do as following:
```bash
MODEL=Qwen/Qwen2.5-7B SHORT=Qwen2.5-7B ./run.sh          # base model
SETTING=ai_ai PREFIX=fac_humor_evil_ai ./run.sh          # AI+AI control
```

## Config (shell vars used below)

```bash
# ── vars (per model; swap MODEL/SHORT for base) ──────────────────────────────
MODEL=Qwen/Qwen2.5-7B-Instruct ; SHORT=Qwen2.5-7B-Instruct      # base: Qwen/Qwen2.5-7B
DATA=persona_steering/data_generation/trait_data_extract
VEC=persona_steering/persona_vectors/$SHORT
EXTRACT=persona_steering/eval_persona_extract/$SHORT

# ── 0. Re-extract BOTH settings' cells WITH per-prompt activations ───────────
#    (only change vs README step 2: add --save_activations)
for PREFIX in fac_humor_evil_user fac_evil_ai_humor_user ; do
  for CELL in pp pn np nn ; do
    python -m persona_steering.eval.eval_persona --model $MODEL --trait ${PREFIX}_${CELL} \
    --output_path $EXTRACT/${PREFIX}_${CELL}_pos_instruct.csv \
    --persona_instruction_type pos --assistant_name ${PREFIX}_${CELL} \
    --version extract --n_per_question 1 --max_tokens 128 --limit_questions 64 --overwrite

    python -m persona_steering.eval.eval_persona --model $MODEL --trait ${PREFIX}_${CELL} \
    --output_path $EXTRACT/${PREFIX}_${CELL}_neg_instruct.csv \
    --persona_instruction_type neg --assistant_name ${PREFIX}_${CELL} \
    --version extract --n_per_question 1 --max_tokens 128 --limit_questions 64 --overwrite

    python persona_steering/generate_vec.py --model_name $MODEL \
      --pos_path $EXTRACT/${PREFIX}_${CELL}_pos_instruct.csv \
      --neg_path $EXTRACT/${PREFIX}_${CELL}_neg_instruct.csv \
      --trait ${PREFIX}_${CELL} --save_dir $VEC \
      --threshold 0 --coherence_threshold 0 --save_activations
  done
  python factorial_residue.py --vec_dir $VEC --prefix $PREFIX --per_prompt   # <-- emits _R + per_prompt_residue
done

# ── 1. Metrics + bootstrap CIs + permutation null (per setting) ─────────────
for PREFIX in fac_humor_evil_user fac_evil_ai_humor_user ; do
  python metrics.py --vec_dir $VEC \
    --ai_trait ${PREFIX}_fac_tauA --user_trait ${PREFIX}_fac_tauU --joint_trait ${PREFIX}_fac_joint \
    --per_prompt_residue $VEC/${PREFIX}_fac_per_prompt_residue.pt \
    --n_boot 2000 --n_perm 2000 \
    --output_dir output/metrics_factorial_$SHORT --output_prefix $PREFIX
done
# -> *_bootstrap_ci.csv, *_permutation_null.csv, *_effrank.csv  (effrank answers the subspace question)

# ── 2. Trait leakage (cross-setting; run after BOTH settings exist) ─────────
python trait_leakage.py \
  --source_vec_dir $VEC --source_name fac_humor_evil_user_fac_tauU \
  --target_vec_dir $VEC --target_name fac_evil_ai_humor_user_fac_tauA \
  --label evil --out_dir output/leakage_$SHORT --emit_shared
# base→instruct headline (once base exists): add
#   --base_source_vec_dir <base VEC> --base_target_vec_dir <base VEC>

# ── 3. Causal trio (the surgical ablations) ─────────────────────────────────
EVIL_R=$VEC/fac_evil_ai_humor_user_fac_R_response_avg_diff.pt
EVIL_AXIS=$VEC/fac_evil_ai_humor_user_fac_tauA_response_avg_diff.pt

# 3a. Does undoing R's evil-suppression INCREASE evil?  (proves R suppresses evil)
python ablation_study.py --model_name $MODEL --modes baseline,restore_along,ablate_R \
  --residue $EVIL_R --restore_target $EVIL_AXIS \
  --coeff "0,1,2,4" --intervention_layers 15-28 \
  --trait_json $DATA/fac_evil_ai_humor_user_pp.json --persona_side pos \
  --n_per_question 1 --instruction_indices 0 \
  --output_dir output/abl_restore_$SHORT --output_csv generations.csv
python score_ablation_outputs.py --input_csv output/abl_restore_$SHORT/generations.csv --answer_col answer --overwrite

# 3b. Does ablating the shared evil axis KILL the user→assistant sway?  (leakage defense)
#     run on the humor-AI/evil-USER condition (Setting A pp) and its ethical-user control (pn)
for C in pp pn ; do
 python ablation_study.py --model_name $MODEL --modes baseline,ablate_vec \
  --ablate_vec $EVIL_AXIS --coeff "1" --intervention_layers 15-28 \
  --trait_json $DATA/fac_humor_evil_user_${C}.json --persona_side pos \
  --n_per_question 1 --instruction_indices 0 \
  --output_dir output/abl_leak_${C}_$SHORT --output_csv generations.csv
 python score_ablation_outputs.py --input_csv output/abl_leak_${C}_$SHORT/generations.csv --answer_col answer --overwrite
done
# sway = evil(pp)-evil(pn); defense works if this difference shrinks under ablate_vec

# 3c. Does ADDING the shared-leak vector inject evil even with an ETHICAL user? (amplify)
python ablation_study.py --model_name $MODEL --modes baseline,add_raw \
  --add_vec output/leakage_$SHORT/evil_shared_leak_response_avg_diff.pt \
  --coeff "0,1,2" --intervention_layers 15-28 \
  --trait_json $DATA/fac_humor_evil_user_pn.json --persona_side pos \
  --n_per_question 1 --instruction_indices 0 \
  --output_dir output/abl_amplify_$SHORT --output_csv generations.csv
python score_ablation_outputs.py --input_csv output/abl_amplify_$SHORT/generations.csv --answer_col answer --overwrite

# ── 4. Plots (per setting) ──────────────────────────────────────────────────
python make_plots.py --vec_dir $VEC --fac_prefix fac_evil_ai_humor_user_fac \
  --metrics_csv output/metrics_factorial_$SHORT/fac_evil_ai_humor_user_per_layer_metrics.csv \
  --model_short $SHORT --base_trait evil_ai_humor_user \
  --refusal_csv output/metrics_factorial_$SHORT/fac_evil_ai_humor_user_refusal_cosines.csv \
  --leakage_csv output/leakage_$SHORT/evil_leakage.csv \
  --out_dir output/plots/$SHORT/evilai
# base→instruct overlay on alpha & leakage: add
#   --base_metrics_csv <base per_layer_metrics.csv> --base_leakage_csv <base evil_leakage.csv>
```

---

## Notes

- Factorial cells **must** use `--threshold 0 --coherence_threshold 0`; the default (50) drops
  different rows per cell and breaks cancellation.
- Per-prompt EffRank (`factorial_residue.py --per_prompt`, `metrics.py --per_prompt_residue`)
  needs `*_activation_matrix.pt` saved with identical row order across all 4 cells; the
  mean-vector pipeline above is sufficient for everything else.
- Cross-model results are compared by **structure** (tensor rank, slot asymmetry, refusal
  routing), not raw values. The last captured layer is a degenerate artifact — exclude it.