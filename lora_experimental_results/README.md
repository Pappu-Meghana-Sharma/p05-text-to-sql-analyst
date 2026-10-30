# LoRA experimental results (Part 4: fine-tuning)

Execution accuracy on the 200-question test and val splits (same splits and `same_result` check as Parts 2/3).
LoRA r=16, alpha=32, q/k/v/o, 1 epoch, lr 2e-4, max_len 512, batch 2x8.

| Model | Train pairs | Test | Val |
|---|---|---|---|
| 0.5B | 0 (base) | 0.535 | 0.50 |
| | 1k | 0.565 | 0.53 |
| | 5k | 0.605 | 0.60 |
| | 10k | 0.655 | 0.615 |
| Coder-1.5B | 0 (base) | 0.56 | 0.53 |
| | 1k | 0.655 | 0.67 |
| | 5k | 0.69 | 0.715 |
| | 10k | **0.715** | **0.735** |
| RAG BM25 k=3 (Part 2) | - | 0.67 | 0.695 |

Files: `lora_metrics_<adapter>_<split>.json` (summary), `lora_eval_<adapter>_<split>.txt` (per example),
`lora_results.csv` (all runs), `lora_train_log.txt` (training loss), `logs/` (raw run logs).
`lora_adapter` = 0.5B 1k; `base` = 0.5B base; `base_coder` = Coder-1.5B base.
