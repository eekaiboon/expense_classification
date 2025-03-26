# Intro

A custom BERT-based model designed to classify expense categories by integrating textual descriptions with numeric features (e.g., transaction amounts). This model extends a standard BERT backbone by appending a numeric feature to the [CLS] embedding, allowing it to leverage both language understanding and numerical insights for accurate expense categorization.

## Features

- **Expense Classification:** Designed specifically for categorizing expenses.
- **BERT-Based Architecture:** Utilizes a pre-trained BERT model to process text.
- **Numeric Feature Fusion:** Combines text embeddings with numeric data for improved classification.
- **Weighted Training:** Incorporates weighted loss to address class imbalance.
- **Modular Design:** Separate modules for data loading, model definition, and training.

### Required Libraries

Install the necessary libraries using pip:

```bash
pip install transformers datasets torch scikit-learn
```