# SentinelNLP

SentinelNLP is a reproducible research prototype for studying cross-lingual
vulnerability-severity prediction and explanation faithfulness across English,
Hindi, Simplified Chinese, Spanish, Arabic, and French.

## Current phase

The MVP foundations now include NVD v2 retrieval/filtering, CVE-level splits,
a TF-IDF word/character baseline, translation import validation, metric and
Critical-risk helpers, and explicit API endpoints. No dataset, model,
translation, or evaluation result is bundled or fabricated.

The prototype does not replace official CVSS assessment or expert review.

## Optional supplementary dataset: SEVENLLM

SEVENLLM is not part of SentinelNLP's primary classification dataset. The
official Hugging Face dataset card describes
[SEVENLLM-Dataset](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/SEVENLLM-Dataset)
as a bilingual English/Chinese cybersecurity incident resource with these
JSONL fields:

```json
{
  "category": "...",
  "instruction": "...",
  "input": "...",
  "thought": "...",
  "output": "..."
}
```

Its official repository metadata reports an Apache-2.0 dataset-card license,
an approximately 91,401-record `train.jsonl`, and a 1,300-record
`test.jsonl`. The records are instruction/question-answer, structured
understanding, generation, and multiple-choice examples rather than
CVSS-labeled vulnerability records. The schema does not provide a required
`cve_id`, CVSS v3.1 vector/base score, or verified severity label.

For that reason, SentinelNLP must keep NVD as the primary dataset and must
not merge SEVENLLM instruction or question-answer records into the CVSS v3.1
classification data. SEVENLLM may be evaluated separately as an optional
cybersecurity-language resource—for example, incident understanding,
information extraction, or multilingual instruction-following—only after a
separately approved experiment. Any future linkage to the primary dataset
must require a reliable CVE identifier match and independently verified NVD
severity label; otherwise the record remains supplementary and excluded from
classification training/evaluation.

The full SEVENLLM dataset is intentionally not downloaded or vendored in this
repository. The dataset card also notes that its fields can contain sensitive
network-security terminology. The dataset license declaration should be
reviewed alongside the provenance and licenses of any underlying source text
before redistribution.

## Local development

### Backend

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.app.main:app --reload
```

Run tests with:

```powershell
pytest backend\tests
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

### Explicit research workflows

To download a small real NVD dataset, run:

```powershell
python scripts\prepare_dataset.py --limit 25 --request-delay 0.7
```

This writes the raw API snapshot to `data\raw\nvd_cves_raw.json`, the
processed CVSS v3.1 records to `data\processed\cves.csv`, and provenance and
filtering metadata to `data\processed\dataset_metadata.json`. Retrieval is
explicit only; application startup does not access the NVD API.

For a larger dataset:

```powershell
python scripts\prepare_dataset.py --limit 5000 --request-delay 0.7
```

The NVD and baseline commands operate on real data and can require network,
disk, and compute resources. XLM-R training intentionally fails with a clear
message unless PyTorch and Transformers are installed in a dedicated GPU
environment; the command downloads `xlm-roberta-base` only when explicitly
run. Application startup never downloads models or launches experiments.

To evaluate severity predictions against the NVD severity label for English and
each available translation language, run:

```powershell
python scripts\evaluate_translations.py `
  --translations data\translations\translations.csv `
  --records data\processed\nvd_records.json `
  --model models\baseline\model.joblib `
  --output artifacts\metrics.json
```

The report contains accuracy, macro/weighted F1, balanced accuracy, confusion
matrices, per-class results, English agreement, and Critical recall for
English, Hindi (`hi`), Simplified Chinese (`zh`), Spanish (`es`), and Arabic
(`ar`) or French (`fr`) when those languages are present. Rows failing
translation validation are excluded and counted in `excluded_rows`; they are
not silently treated as valid evaluation samples. The current artifact has 19
valid rows per target language and excludes 5 rows with missing technical
identifiers.

When `models\xlmr_severity` contains a trained XLM-R model, the API loads its
tokenizer and model once during application startup using local files only. The
`/api/status` and `/api/models` endpoints report XLM-R as available only after
that load succeeds. If PyTorch, Transformers, or the model files are missing,
the baseline API remains available and XLM-R requests return HTTP 503 with the
load failure reason. Install the transformer dependencies from
`backend\requirements.txt` before enabling XLM-R inference.

### Valid baseline experiment

The reproducible baseline uses `data\splits\cve_splits.json`, whose version 2
manifest groups identical normalized English descriptions before assigning
train, validation, and test records. Run these commands from the project
root:

```powershell
python scripts\train_baseline.py
python scripts\evaluate_baseline.py
python scripts\evaluate_translations.py --translations data\translations\translations.csv
```

The model is fit only on the manifest's training CVE IDs. Held-out
predictions are saved to `artifacts\baseline_test_predictions.csv`, primary
metrics to `artifacts\baseline_metrics.json`, and model provenance to
`models\baseline\metadata.json`. The existing translation file was generated
for the previous split; rows outside the current test split are excluded and
the resulting artifact is marked partial. It must not be described as a
complete multilingual test evaluation.

Because paid or network translation services are not part of the reproducible
workflow, `scripts\prepare_translation_import.py` creates
`data\translations\current_test_translations.csv` as a manual/external import
template. It creates one row for each current test CVE and target language,
reuses only source-matched non-placeholder translations, and leaves missing
translations explicitly excluded. The current repository has no reliable
non-English translations for this grouped test split, so multilingual metrics
remain blocked rather than fabricated.

### Local machine translation workflow

The expanded experiment includes a resumable local translation command:

```powershell
python scripts\translate_expanded_test.py `
  --records data\expanded\nvd_records.json `
  --splits data\expanded\cve_splits.json `
  --output data\translations\expanded_test_translations_generated.csv `
  --limit-per-language 5
```

It uses `facebook/nllb-200-distilled-600M` with the verified NLLB codes
`hin_Deva`, `zho_Hans`, `spa_Latn`, `arb_Arab`, and `fra_Latn`. The model is
CC-BY-NC-4.0, requires approximately 2.5 GB of weights, and is CPU-intensive.
The generated file is separate from the blank import template and records the
dataset/split hashes, model revision, generation settings, dependency versions,
and validation status. Use `--retry-failed` to retry rejected rows; valid rows
are skipped.

The generator checkpoints after each language batch, records source/output
token counts and truncation flags, and retries bounded runtime/value failures.
Rows are resumable only when their source, language, and current model
revision match the generated artifact. The confirmed pipeline audit is saved
in `artifacts\translation_pipeline_audit.json`.

The validator distinguishes `valid`, `needs_review`, `failed`, and `missing`.
Technical checks cover CVE IDs, versions, product/symbol names, paths,
filenames, hashes, and related identifiers. A missing identifier is reported
with its category; it is never repaired by appending text to a translation.
Automated validation is separate from human semantic review, and only
validated rows may enter multilingual metrics.

Before a full run, use `scripts\benchmark_translation.py` on at least three
current-test CVEs with varied lengths and severities. The bounded CPU results
and exact local model revision are recorded in
`artifacts\translation_benchmark_summary.json`. The measured CPU estimate is
too long for an unapproved full run; use a faster machine, a suitable local
GPU, or an explicitly approved translation source instead. No vulnerability
text is sent to an external service by these scripts.

Validate and evaluate generated rows separately:

```powershell
python scripts\validate_translations.py `
  --translations data\translations\expanded_test_translations_generated.csv `
  --output artifacts\expanded_translation_validation.json

python scripts\evaluate_translations.py `
  --translations data\translations\expanded_test_translations_generated.csv `
  --records data\expanded\nvd_records.json `
  --splits data\expanded\cve_splits.json `
  --model models\expanded_baseline\model.joblib `
  --output artifacts\expanded_multilingual_metrics.json `
  --predictions-output artifacts\expanded_multilingual_predictions.csv
```

Generated translations are machine-translated and are not human-verified.
Rows are eligible for evaluation only after exact source matching, current
test-split membership, supported-language checks, placeholder rejection, and
technical-identifier preservation. The first CPU smoke pilot generated four
eligible rows and stored one stale Hindi rejection caused by sentence
punctuation in the identifier matcher; the corrected validator accepts all
five pilot rows. No full 1,625-pair run was started.

For the bounded 25-CVE pilot, the Phase 2 audit and coverage artifacts can be
recreated without translation generation:

```powershell
python scripts\audit_translation_pilot.py `
  --translations data\translations\expanded_translation_pilot_25.csv

python scripts\validate_translations.py `
  --translations data\translations\expanded_translation_pilot_25.csv `
  --output artifacts\expanded_translation_pilot_25_validation_revised.json

python scripts\prepare_translation_review.py `
  --translations data\translations\expanded_translation_pilot_25.csv `
  --output artifacts\translation_pilot_25_review_stratified.csv
```

The audit retains the complete source and generated text for every rejected or
suspicious row in `artifacts\translation_pilot_failure_analysis.json` and
records strictly matched five-language coverage in
`artifacts\translation_pilot_matched_coverage.json`. Identifier protection is
used only for new or explicitly retried generation. Internal placeholder slots
are checked across chunk boundaries, but are not sent to NLLB: the model
repeatedly dropped or altered visible markers even when those markers had
round-tripped through its tokenizer. The translation pipeline translates only
the surrounding text and restores each original identifier into its verified
source slot; angle-bracket path variables are protected occurrence-by-
occurrence, and punctuation and whitespace at span boundaries stay outside
the translated text. Missing, duplicated, unknown, fused, or damaged
identifiers reject the row. Each generated text span has its own output-token
cap and must end with the model's EOS token; a missing completion marker
rejects the row. Chinese shortness checks count Han characters rather than
whitespace-delimited words. These exact-preservation and completion checks do
not establish semantic translation quality. Existing pilot output is never
rewritten, and the review worksheet remains `unreviewed` until a person labels
each row.

The 22-pair failure-recovery set has a separate review-ready worksheet at
`artifacts\translation_human_review_worksheet_gpu_v5.csv` and a predeclared
rubric and scale-up gate at
`artifacts\translation_human_review_protocol_v1.md`. Only qualified
target-language reviewers should enter ratings. The worksheet distinguishes
human review from automated integrity checks and leaves all human ratings
blank until reviewers supply them. After review data is entered, validate and
summarize it with:

```powershell
python scripts\summarize_translation_human_review.py `
  --worksheet artifacts\translation_human_review_worksheet_gpu_v5.csv `
  --output artifacts\translation_human_review_summary_v1.json
```

The summary reports unreviewed, incomplete, accepted, and rejected rows;
incomplete entries are never counted as passes. Use `--allow-incomplete` only
for a clearly labeled progress report. The script derives acceptance from the
protocol rubric and keeps the full translation grid NO-GO unless the stated
human-review gate is met. These 22 targeted failure-recovery pairs are not a
representative random sample and cannot establish the quality of all 1,625
planned translations.

For a bounded retry, `scripts\prepare_translation_retry.py` writes the exact
audited target-pair manifest. The translation generator accepts that manifest,
requires CUDA when `--device cuda` is specified, supports local-only pinned
model loading, and checkpoints after each generated batch:

```powershell
python scripts\prepare_translation_retry.py

python scripts\translate_expanded_test.py `
  --pairs-file artifacts\translation_pilot_retry_targets.csv `
  --output data\translations\expanded_translation_pilot_25_retry_gpu.csv `
  --revision f8d333a098d19b4fd9a8b18f94170487ad3f821d `
  --device cuda `
  --local-files-only `
  --retry-failed
```

The retry output and its sidecar manifest remain separate from the original
pilot CSV. GPU memory, model-load/generation timing, and CUDA use are recorded
in the generated manifest; a targeted validation/comparison should be run
before promoting any machine translation, and human semantic review remains
required.

To build the non-destructive research audit and metrics bundle for the
expanded experiment, run:

```powershell
python scripts\research_metrics.py `
  --faithfulness artifacts\expanded_faithfulness_recomputed.json
```

This writes separate reports under `artifacts\` for English classification,
Critical-risk metrics, English LIME perturbation diagnostics, multilingual
coverage, cross-language consistency, RQ3 statistical status, provenance, and
a CSV metric-status table. It does not retrain models, alter split manifests,
or count missing/failed translations as predictions. Target-language and RQ3
reports remain blocked until validated matched-CVE translations and
predictions exist.

### Expanded NVD experiment

An expanded, separate NVD snapshot can be collected without overwriting the
original experiment:

```powershell
python scripts\prepare_dataset.py `
  --limit 2000 `
  --start-date 2024-01-01T00:00:00Z `
  --end-date 2024-04-01T00:00:00Z `
  --page-size 200 `
  --request-delay 0.7 `
  --output data\expanded\cves.csv `
  --raw-output data\expanded\nvd_cves_raw.json `
  --metadata-output data\expanded\dataset_metadata.json
```

The expanded run produced 2,163 eligible CVSS v3.1 records from 2,200
retrieved NVD items. Its grouped split, model, predictions, baseline metrics,
faithfulness, and blocked multilingual artifacts are stored under
`data\expanded`, `models\expanded_baseline`, and `artifacts\expanded_*`.
The expanded test translation template contains 1,625 blank rows
(325 CVEs x 5 languages); no translation service is called by the repository.

### Leakage-safe XLM-R evaluation

`scripts\train_xlmr.py` trains a fresh XLM-R checkpoint from the pinned
`FacebookAI/xlm-roberta-base` revision using only the expanded split manifest's
training CVEs. It uses the validation CVEs for macro-F1 checkpoint selection
and early stopping, records split CVE IDs and hashes, and refuses to overwrite
a non-empty experiment directory. For example:

```powershell
python scripts\train_xlmr.py `
  --data data\expanded\nvd_records.json `
  --splits data\expanded\cve_splits.json `
  --output artifacts\xlmr_english_20261026_v2 `
  --model FacebookAI/xlm-roberta-base `
  --revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089
```

For a validation-motivated class-weighted follow-up, pass
`--class-weighting inverse_sqrt`; weights are derived from training counts
only, and per-class validation precision, recall, and F1 are logged each
epoch.

After training, `scripts\evaluate_xlmr_research.py` evaluates English and
recomputed-valid translations on the same fixed test CVEs, reports bootstrap
intervals and paired baseline comparisons, and writes predictions and metrics
to a separate experiment directory. Translation validation is automated;
machine translations remain unreviewed unless separately marked as human
reviewed. The XLM-R workflow requires CUDA-enabled PyTorch, Transformers, and
Accelerate in the selected Python environment.

For a leakage-safe multilingual run, first create a `cve_ids` JSON and an exact
`cve_id,language` CSV pair manifest containing only fixed train and validation
CVEs, then run `scripts\translate_expanded_test.py` with
`--split-scope train_validation`, those manifests, and a separate resumable
output CSV. Pin `--model facebook/nllb-200-distilled-600M` to the recorded
NLLB revision and generation settings. Pass the resulting CSV to
`scripts\train_xlmr.py --multilingual-translations`; only nonempty rows with
matching source text, severity, source hash, and recomputed automated
validation status are admitted. Failed and missing pairs are recorded in
`translation_validation_report.json`. To compare previous runs on the same
valid test rows, pass their `predictions` folders using repeated
`--comparison-prediction-dir` arguments to `scripts\evaluate_xlmr_research.py`.
When previous test results have already been observed, add
`--exploratory-test-comparison` so the evaluation artifacts explicitly record
that interpretation.
