# LLM-Agent-Evaluation-Robustness-Generalization-and-NLP-Based-Assessment

NLP-based evaluation of LLM-agent responses using duplicate-controlled testing, cross-domain generalization, and statistical analysis.

## Overview

This project investigates how NLP methods can classify failure types in LLM-agent responses and how performance changes under different evaluation settings.

The analysis uses a **synthetic benchmark of 1,500 agent-task interactions** across five domains: Coding, Mathematics, RAG/QA, Planning, and Customer Support, covering 12 failure categories and 4 severity levels.

The project examines repeated responses, duplicate-controlled evaluation, feature contributions, cross-domain generalization, classification errors, lexical similarity, and statistical differences across failure categories.

## Research Questions

1. How accurately can NLP models classify LLM-agent failure types?
2. How does performance change after controlling for repeated responses?
3. How well does the classifier generalize to unseen domains?
4. Which textual inputs contribute most to classification?
5. Does expected-vs-agent answer similarity differ across failure types or severity?

## Methodology

- Exploratory analysis of domains, failure types, severity, difficulty, and text characteristics
- Synthetic-data and duplication audit
- TF-IDF-based NLP classification using Logistic Regression
- Word, character, and combined TF-IDF comparisons
- Feature ablation using prompt, context, and agent answer
- Leave-one-domain-out evaluation
- Classification error analysis
- TF-IDF cosine similarity between expected and agent answers
- Kruskal-Wallis and post-hoc Mann-Whitney U tests with Holm correction

## Results

### Classification Performance

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Random Split | 1.0000 | 1.0000 |
| Duplicate-Controlled | 0.9367 | 0.9208 |
| Leave-Coding-Out | 0.2900 | 0.2632 |
| Leave-Mathematics-Out | 0.4267 | 0.4502 |
| Leave-RAG/QA-Out | 0.3633 | 0.3603 |
| Leave-Planning-Out | 0.4933 | 0.3902 |
| Leave-Customer-Support-Out | 0.4467 | 0.3838 |

The perfect random-split result was treated cautiously because the dataset contains repeated agent responses. Duplicate-controlled evaluation produced 93.67% accuracy and 92.08% Macro-F1.

### Feature Comparison

| Input | Accuracy | Macro-F1 |
|---|---:|---:|
| Prompt + Context + Agent Answer | 0.9367 | 0.9208 |
| Agent Answer Only | 0.8987 | 0.7214 |

Prompt and context information improved classification performance compared with using the agent answer alone.

### Statistical Analysis

Lexical similarity differed significantly across failure types:

**Kruskal-Wallis H = 59.54, p < 0.001**

Post-hoc Mann-Whitney U tests with Holm correction identified significant differences among several failure-type pairs.

No statistically significant difference in lexical similarity was detected across severity levels:

**Kruskal-Wallis H = 2.63, p = 0.453**

## Key Findings

- Random-split performance was likely optimistic because of repeated synthetic responses.
- Duplicate-controlled classification remained strong at 93.67% accuracy and 92.08% Macro-F1.
- Cross-domain performance was substantially lower than within-benchmark performance.
- Prompt and context information contributed more than the agent answer alone in the tested feature setting.
- Lexical similarity differed significantly across failure categories.
- No statistically significant difference was detected across severity levels.

## Limitations

- The dataset is synthetic and benchmark-specific.
- The original dataset contains repeated agent responses.
- The duplicate-controlled analysis contains 392 unique agent answers.
- Several failure categories have small sample sizes.
- TF-IDF cosine similarity measures lexical overlap rather than semantic equivalence.
- Cross-domain results are specific to the five benchmark domains.
- Results should not be interpreted as estimates of real-world failure frequencies or production-agent reliability.
## Dataset Source

- Dataset size: 1,500 synthetic examples
- Domains: Coding, Mathematics, RAG/QA, Planning, Customer Support
- Failure categories: 12
- Dataset license: Apache License 2.0
- Raw dataset: Not redistributed in this repository

This project focuses on independent analysis and evaluation of the dataset, including duplicate-controlled testing, NLP classification, cross-domain generalization, feature ablation, error analysis, and statistical analysis.

This project uses the [AI Agent Failure Benchmark Dataset](https://www.kaggle.com/datasets/sunil123kumar/ai-agent-failure-benchmark-dataset) created by Sunil Kumar and published on Kaggle.


## Reproducibility

The complete analysis is provided in:

`llm_agent_evaluation.ipynb`

### Requirements

```text
pandas
numpy
scikit-learn
scipy
statsmodels
matplotlib
seaborn
jupyter
