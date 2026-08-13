# Model placement

Trained PyTorch files live here:

```
backend/models/classification/facial_encoder.pt         # training/train_facial.py, on dataset/Extracted_images
backend/models/classification/speech_encoder.pt         # training/train_speech.py, on dataset/Audios
backend/models/classification/mental_health_classifier.pt  # training/train_fusion.py
backend/models/regression/depression_regressor.pt          # training/train_fusion.py
backend/models/regression/anxiety_regressor.pt              # training/train_fusion.py
backend/models/regression/stress_regressor.pt                # training/train_fusion.py
backend/models/fusion/multimodal_fusion.pt                   # training/train_fusion.py (trunk only)
```

These files **are committed**, so a clone runs the real models with no training step.

`speech_encoder.pt` is stored in a slim format written by
`training/slim_speech_checkpoint.py`: only the fine-tuned transformer layers and head, since the
frozen wav2vec2-base weights are identical to what `from_pretrained()` fetches at construction.
That takes it from 360MB (over GitHub's 100MB limit) to 54MB, bit-identical — verified at a maximum
logit difference of `0.000e+00`. The loader detects the format via the `__mindscan_slim__` marker and
completes it from the HuggingFace backbone, so the **first run needs internet**; it is cached after.

`registry.using_mock` (backend/core/model_loader.py) goes `false` once both encoder files are present and
`USE_MOCK_INFERENCE=false` is set — that switches the facial/speech embeddings used by fusion and the
Grad-CAM/LIME explainers over to the real trained encoders. The classifier/regressors are gated
independently: if their four files are present (with `USE_MOCK_INFERENCE=false`),
`registry.predict_status()`/`registry.predict_scores()` return real predictions; otherwise they return
`None` and `backend/api/routes/assessment.py` falls back to the heuristic engine
(`classify_status`/`estimate_scores`) directly. The response's `using_trained_tabular_model` field reports
which path a given request actually used.

**Important caveat on the classifier/regressors**: `dataset/mental_health_multimodal.csv` (the original
4000-row, 18-feature file) was checked for a learnable relationship between its features and
`Mental_Health_Status`/D-A-S scores two ways — linear correlation (all |r| < 0.05) and a 300-tree random
forest (39.3% accuracy vs. a 40.7% majority-class baseline; per-feature importances uniform at ~1/16) —
and neither found signal above the trivial baseline: a model trained directly on it just fits noise.
`training/train_fusion.py` instead trains on `dataset/mental_health_multimodal_synthetic_labels.csv`,
produced by `training/generate_synthetic_labels.py`: the same 18 real feature columns, with
`Mental_Health_Status`/D-A-S scores **regenerated** from the app's own heuristic formulas
(`backend/core/inference/estimate_scores`) plus Gaussian noise, so a genuine (if designed) relationship
exists to learn. **This is not real clinical ground truth.** The deployed classifier/regressors
demonstrate the training pipeline works when a real feature-label relationship exists, and approximate
the app's own heuristic (with realistic noise/generalization error) — they do not carry validated
psychological predictive power. Test-set results (held-out 600 rows, stratified):

| Model | Metric | Result | Baseline |
|---|---|---|---|
| Status classifier | accuracy / balanced acc / macro-F1 | 47.7% / 39.1% / 34.2% | 40.7% majority-class |
| Depression regressor | MAE | 3.41 | 5.58 (mean-predict) |
| Anxiety regressor | MAE | 2.67 | 3.66 (mean-predict) |
| Stress regressor | MAE | 4.28 | 4.66 (mean-predict) |

The classifier's macro-F1 is noticeably weaker than its accuracy — it struggles most on `Mild_Stress`,
the class squeezed between `Healthy` and `Moderate_Stress` on the underlying continuous burden score,
where the injected noise blurs the boundary most.

Input specs (from the Hack2Health dataset description):

| Stream | Tensor | Encoder |
|---|---|---|
| Facial | `(1, 1, 48, 48)` grayscale float32 0–1 (FER) | ImageNet-pretrained ResNet18, fine-tuned (`training/train_facial.py`) — 70.3% val acc, 7-class FER |
| Speech | `(N,)` raw mono waveform, 16kHz float32; optional RAVDESS filename metadata | Pretrained wav2vec2-base, last 2 transformer layers fine-tuned + trained attention-pool/embed head (`training/train_speech.py`) — 67.08% test acc, unseen-speaker split, 4-class RAVDESS_TO_STATUS (Healthy/Mild/Moderate/Severe) |
| Numerical | `(18,)` z-scored columns listed in `docs/dataset.md` | `FusionTrunk` + `StatusClassifier`/`ScoreRegressor` heads (`training/train_fusion.py`) on synthetic labels, heuristic fallback — see above |

Labels: `Healthy`, `Mild_Stress`, `Moderate_Stress`, `Severe_Stress`.
Scores: depression 0–34, anxiety 0–24, stress 0–39.
