# Contributing to RecommendKit

Thanks for your interest in contributing to **RecommendKit** 👋

This document explains what the toolkit is, how to get started, and the main areas where contributions are especially welcome.

---

## What is RecommendKit?

**RecommendKit** is a **universal, end-to-end recommendation system framework**.

It's not just boilerplate training code — it's a **ready-to-use system** that takes you from raw user–item data to a trained, persisted recommendation model with minimal configuration.

**Key properties:**

* **Universal** – Works across domains: e-commerce, content, social, jobs, etc.
* **End-to-end** – Includes:
  * A simple JSON-based data format for users, items, and interactions
  * Automatic positive/negative sample generation (with no data leakage)
  * Training pipeline with train/val/test splits
  * Inference for embeddings, similarity scores, and top-K recommendations
* **Config-driven** – Most behavior is controlled via JSON configs.
* **Flexible architecture** – Two-tower design with:
  * **SimpleFusion** (fast concat + MLP) as the default
  * **Attention-based fusion** for richer, multi-modal feature sets
* **Extensible** – Plug in your own encoders, interaction models, and heads across modalities.

For a deeper dive into the **project structure** and **architecture**, see the `README.md` sections:

* 📁 **Project Structure**
* 🏗️ **Architecture** (Two-Tower, SimpleFusion, Attention-Based Fusion)
* 🔧 **Customization**

---

## Getting Started

Before contributing, it's helpful to get familiar with the system end-to-end.

### 1. Run the Quickstart Notebook

The best entry point is:
```
jupyter notebook quickstart.ipynb
```

The notebook walks you through:

* Loading sample / synthetic data
* Understanding the JSON data format (users, items, interactions)
* Training a recommendation model using a provided config
* Running inference to get embeddings, scores, and top-K recommendations

If you're new to the repo, running this notebook once is highly recommended.

---

### 2. Explore the Codebase & Architecture

Use the **Project Structure** section in `README.md` as your map. In particular:

* `configs/`
  * `correlated_dataset_config.json` – SimpleFusion config
  * `correlated_dataset_attention_config.json` – attention-based config
* `datasets/synthetic/` – Synthetic dataset generator and sample JSON dataset
* `encoders/` – Encoders for text, image, categorical, continuous, temporal features
* `interaction/` – Feature fusion (SimpleFusion, attention) and interaction modeling
* `classifier/` – Classification heads and losses
* `trainer/` – Training pipeline and data loading
* `train.py` – Main training entrypoint (includes train/val/test split)
* `inference.py` – Inference script for embeddings and recommendations

Also read the **Architecture** section in the README to understand:

* The two-tower (user/item) setup
* How SimpleFusion vs. attention-based fusion is configured
* How configs toggle fusion modes without code changes

---

## How to Contribute

We follow a standard GitHub workflow.

### 1. Fork the Repository

On GitHub, click **Fork**, then clone your fork:
```
git clone https://github.com/<your-username>/RecommendKit.git
cd RecommendKit
```

Set up your environment following the instructions in `README.md` (dependencies, Python version, etc.).

---

### 2. Create a Feature Branch

Use a descriptive branch name:
```
git checkout -b feature/your-feature-name
```

Examples:

* `feature/add-attention-metrics`
* `feature/vector-db-support`
* `chore/add-new-text-encoder`

---

### 3. Make Your Changes

* Implement your feature / bugfix / documentation update.
* Keep code style consistent with the existing codebase.
* If you add new configs or encoders, try to keep them as self-explanatory as the existing ones.

---

### 4. Run Checks

Before opening a PR, sanity-check the main flows. For example:
```
# Train using the default config
python3 train.py \
  --config configs/correlated_dataset_config.json \
  --data datasets/synthetic/correlated_dataset.json
```

If your change affects inference, also test:
```
python3 inference.py \
  --model_path models/your_trained_model.pth \
  --config configs/correlated_dataset_config.json \
  --data test_input.json
```

If `quickstart.ipynb` or other notebooks rely on your changes, run them or at least re-execute the affected cells.

---

### 5. Commit, Push, and Open a PR
```
git add .
git commit -m "Short description of your change"
git push origin feature/your-feature-name
```

Then open a **Pull Request** against the main `RecommendKit` repo.

In the PR description, please include:

* A summary of what you changed
* Any new configs / flags / encoders introduced
* How you tested it (commands, datasets, notebooks)

We'll review, give feedback if needed, and merge once it's ready ✅

---

## High-Impact Contribution Areas

Here are specific tasks where help would be especially valuable.

### 1. Test Attention Fusion on a Real Multi-Modal Dataset

**Goal**  
Evaluate the **attention-based fusion configuration** on a real item–user dataset with multiple modalities (e.g., text + images + categorical features).

**What this might involve**

* Finding or building a dataset with:
  * User features
  * Item features across different modalities
  * Interaction logs (clicks, purchases, etc.)
* If web scraping is involved, please respect:
  * `robots.txt`
  * Site terms of service
  * Reasonable rate limits
* Wiring this dataset into the existing training pipeline via a config.
* Running experiments:
  * SimpleFusion vs. attention-based fusion
* Summarizing observations in:
  * A markdown note under `docs/`, and/or
  * A dedicated example notebook

---

### 2. Add Evaluation Metrics on the Test Split

`train.py` already handles **train/val/test** splitting.

**Goal**  
Integrate standard ranking metrics computed on the **test** set:

* Precision@k
* Recall@k
* NDCG@k

**What this might involve**

* Adding a small evaluation module (e.g., `metrics.py` or logic under `trainer/`).
* After training completes, run evaluation on the test split and:
  * Log metrics clearly (stdout and/or a structured object).
  * Support configurable `k` values via config or CLI.
* Making the evaluation helper easily reusable from notebooks and scripts.

---

### 3. Add Vector Database Support and Verify Parity

**Goal**  
Integrate a **vector database** so that user/item embeddings produced by RecommendKit can be stored and queried for nearest neighbors, and verify that retrieval results are consistent with the current in-memory flow.

**What this might involve**

* Designing a small abstraction for:
  * Writing embeddings to a vector DB
  * Querying top-K nearest neighbors given a user (or item) embedding
* Building a minimal example flow:
  1. Train a model with `train.py`.
  2. Export embeddings (via `inference.py` or a helper).
  3. Index them into a vector DB.
  4. Compare retrieved results vs. RecommendKit's in-memory similarity computation.
* Documenting the integration in a README section or notebook.

---

### 4. Add More Encoder Models Across Modalities *(Great First Contribution!)*

**Goal**  
Expand the library of encoders to make RecommendKit even more plug-and-play.

**Ideas**

* Additional **text** encoders (small transformers, RNN/CNN-based, etc.)
* More **image** encoders (lightweight CNNs, mobile-friendly backbones)
* Alternate **categorical / continuous** encoders (different embedding/normalization strategies)
* Early versions of **audio** or **time-series** encoders if relevant

**What this might involve**

1. Implement the encoder in `encoders/` following existing interfaces.
2. Wire it into the config system so it can be enabled via JSON.
3. Add a small example config and (optionally) a short note or notebook showing how to use it.

This is one of the easiest ways to learn the codebase while making the toolkit more powerful for others.

---

## Questions, Ideas, or Larger Proposals?

If you're unsure where to start or have an idea that goes beyond the list above:

* Open a **GitHub Issue** to start a discussion, or
* Create a **Draft PR** if you already have some code and want early feedback.

Thank you for helping improve RecommendKit 💙  
Every contribution makes this end-to-end recommendation system framework more useful and accessible for everyone.