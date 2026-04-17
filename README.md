# Exploratory Multi-Method Sentiment Analysis on Heterogeneous Sources

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)
![MongoDB](https://img.shields.io/badge/MongoDB-NoSQL-47A248)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-Data%20Processing-E25A1C)

## Project Context & Objectives

Sentiment analysis remains a crucial task in data science, but real-world data is highly heterogeneous. Short tweets, long product reviews, and noisy social posts exhibit vastly different linguistic and statistical properties. 

This project is an exploratory comparative study of binary sentiment classification across four very different datasets:
* Twitter (Sentiment140)
* Amazon Product Reviews (Industrial & Scientific)
* Steam Ubisoft Reviews
* Social Mixed (Multi-platform)

Instead of framing the project as a single-model benchmark, we compare four paper-inspired method families under a shared data pipeline (MongoDB + Spark) and a strict, reproducible 32K evaluation protocol (32,000 training samples / 6,400 evaluation samples). The goal is to understand how different methodological assumptions interact with data properties (length, duplication, class balance, domain shift).

---

## Implemented Methods (Notebooks)

We implemented four distinct approaches, each inspired by a specific research paper. Below is the description of each pipeline and its performance across our datasets under the 32K benchmark.

### 1. HUB-Inspired Social-Sentiment Modeling (DistilRoBERTa)
**Description:** Inspired by the *When Sentiment Analysis Meets Social Network* (HUB) framework. While our datasets lack a usable social graph, we adapted the sentiment-analysis motivation using a robust, fine-tuned `distilroberta-base` Transformer. This acts as our primary, state-of-the-art contextual baseline.

| Scope | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Global (All Sources)** | **0.8447** | **0.8570** | **0.8275** | **0.8420** |
| Twitter | 0.8361 | 0.8402 | 0.8300 | 0.8351 |
| Amazon Reviews | 0.9794 | 0.9910 | 0.9675 | 0.9791 |
| Steam Ubisoft | 0.8942 | 0.8861 | 0.9047 | 0.8953 |
| Social Mixed | 0.5114 | 0.5906 | 0.0744 | 0.1321 |

> Takeaway: Strongest global method. Highly competitive on long texts but collapses entirely on the highly duplicated, low-diversity "Social Mixed" dataset.

### 2. BERT-BiLSTM-Attention Architecture
**Description:** Inspired by Li et al. (2024), this hybrid architecture stacks contextual embeddings (BERT), sequential modeling (BiLSTM), and an Attention mechanism. It is designed to capture long-distance dependencies and explicitly weigh sentiment-bearing words, representing a heavier sequence-modeling design.

| Scope | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Global (All Sources)** | **0.8191** | **0.8319** | **0.7997** | **0.8155** |
| Twitter | 0.8139 | 0.8345 | 0.7831 | 0.8080 |
| Amazon Reviews | 0.9254 | 0.9718 | 0.9366 | 0.9539 |
| Steam Ubisoft | 0.8708 | 0.8721 | 0.8691 | 0.8706 |
| Social Mixed | 0.5222 | 0.5198 | 0.9483 | 0.6716 |

> Takeaway: Second globally, but far more robust on the unstable "Social Mixed" dataset than the pure DistilRoBERTa pipeline, likely due to explicit sequence and attention modeling.

### 3. Naive Bayes & Decision Tree (Steam-Style Baseline)
**Description:** Inspired by Zuo's study on Steam reviews, this pipeline applies classical supervised machine learning methods (NB/DT) with explicit feature engineering (TF-IDF) and rigorous lexical preprocessing. It serves as our low-complexity, interpretable baseline.

| Scope | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Global (All Sources)** | **0.6826** | **0.6780** | **0.6826** | **0.6777** |
| Twitter | 0.6644 | 0.6646 | 0.6644 | 0.6642 |
| Amazon Reviews | 0.8063 | 0.8086 | 0.8063 | 0.8059 |
| Steam Ubisoft | 0.8063 | 0.8086 | 0.8063 | 0.8059 |
| Social Mixed | 0.4837 | 0.4755 | 0.4837 | 0.4655 |

> Takeaway: Weaker globally, but maintains non-trivial scores on Steam and Amazon, proving that classical sparse/lexical signals still hold value on explicit-review domains.

### 4. Recommendation-Justification Transfer
**Description:** Inspired by Ni et al. (2019). The original paper focuses on generating concise explanations from reviews. While the task differs from pure binary sentiment classification, its aspect-aware text modeling offers a transferable perspective on opinion-bearing language. 

| Scope | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Global (All Sources)** | **0.5335** | **0.5337** | **0.5335** | **0.5328** |
| Twitter | 0.8253 | 0.8259 | 0.8253 | 0.8252 |
| Amazon Reviews | 0.9378 | 0.9378 | 0.9378 | 0.9378 |
| Steam Ubisoft | 0.8659 | 0.8720 | 0.8659 | 0.8653 |
| Social Mixed | 0.5152 | 0.5202 | 0.5152 | 0.4838 |

> Takeaway: Suffers globally due to task mismatch (generation vs. classification). Included as an informative negative transfer case, though it performs surprisingly well on specific subsets when evaluated individually.

---

## Prerequisites & Architecture

This project uses a unified Data Engineering and Machine Learning architecture:
* MongoDB (Local): Unified NoSQL storage for the 2.3M+ raw and normalized documents.
* Apache Spark (Local Mode): Standardized text cleaning (URL removal, mention stripping, whitespace normalization) to prevent source-specific preprocessing drift.
* Hugging Face / PyTorch: Used for training the deep learning models (DistilRoBERTa, BERT).
* Scikit-Learn: Used for classical baseline modeling and data splitting.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/siestanba/BigData-Project.git
   ```

2. Install dependencies:  
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

## Database Setup

Ensure MongoDB is running locally on `mongodb://localhost:27017`.  
Import the normalized data into the database `bd_team_normalized`, collection `normalized_reviews`.

## Running the Notebooks

Open Jupyter Notebook or Google Colab and run the scripts located in the `/notebooks` folder.

```bash
jupyter notebook
```

Note: For Deep Learning notebooks, a CUDA-enabled GPU (or Apple MPS) is highly recommended. The 32K benchmark takes approximately 40 minutes per source on a standard accelerator.

## Authors

* Sebastian Straut  
* Nicolas Adamczyk  
* Feng Xiangrui  
* Fallou Diouf  

University of Paris City - Big Data Project
