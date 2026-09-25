# Investigating the Role of Local Context in Named Entity Recognition

Code accompanying my Master's thesis, *"Investigating the Role of Local Context in Named Entity Recognition: Do NER Models Attend to the Entity or the Context?"*, MA Linguistics, University of Cologne, supervised by Prof. Dr. Nils Reiter.

## Research question

Transformer-based NER models are usually evaluated only on whether their final predictions are correct. This leaves open a more basic question: when a model correctly tags an entity, is it actually relying on the entity's own surface form, or is it relying on the surrounding context? This project answers that question empirically, using two complementary approaches:

1. **Masking experiments**: systematically removing either the entity tokens or the context tokens from an input sentence (replacing them with `[MASK]`) and measuring how much predictive accuracy survives in each condition.
2. **Attribution methods (SHAP, LIME)**: generating token-level attribution scores for individual predictions to see which tokens the model's output actually depended on, and comparing these patterns against the masking results.

## Data and model

- **Dataset**: [WNUT-17](https://huggingface.co/datasets/leondz/wnut_17) (Emerging and Rare Entity Recognition), loaded via Hugging Face `datasets`.
- **Model**: [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER), a BERT-based NER model fine-tuned on CoNLL-2003, used here as a pretrained off-the-shelf tagger (not fine-tuned further for this project).

## Repository structure

- **`Main experiment.ipynb`** — the core masking experiment:
  - Overall and per-entity-type accuracy with bootstrapped 95% confidence intervals.
  - Entity-masking vs. context-masking conditions, compared against the unmasked baseline.
  - A probing suite covering ambiguous, unseen, and contradictory examples.
  - Illustrative qualitative examples (original vs. masked predictions).
  - Span-level F1 evaluation using `seqeval`.
  - Sampling-stability checks across random subsets of the test set.
- **`SHAP and LIME.ipynb`** — token-level attribution analysis:
  - A prediction wrapper around the NER pipeline compatible with SHAP and LIME.
  - SHAP explanations using `shap.maskers.Text` with the model's own tokenizer.
  - LIME explanations using `LimeTextExplainer`.
  - Comparison of attribution patterns across the same original / entity-masked / context-masked conditions used in the main experiment.

## Running the notebooks

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run `Main experiment.ipynb` first. It downloads the WNUT-17 dataset and the `dslim/bert-base-NER` model automatically from Hugging Face, and creates a `results/` directory to store intermediate outputs (accuracy summaries, per-type breakdowns, probing results) as CSV/JSON files. These are read back in later cells in the same notebook, so run cells in order top to bottom.
3. Run `SHAP and LIME.ipynb` separately. It loads the same model independently and does not depend on `results/` from the first notebook.

Both notebooks were developed and run in a standard Jupyter environment; no GPU is required, though one will speed up inference over the full test set.

## Notes

- Results are seeded (`RNG_SEED = 42`) for reproducibility of subset sampling and bootstrap confidence intervals.
- The `results/` directory is not included in this repository since it is generated on first run.
- This is thesis research code rather than a packaged library: the notebooks are meant to be read and run in order rather than imported as modules.

## Author

Gonçalo Gonçalves — [ggoncalves956@gmail.com](mailto:ggoncalves956@gmail.com)
