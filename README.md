# Attribute Equivalency Checker
Checker for Correctness of Same File- and Attribute Naming Convention Using a Local LLM

## Overview

This project presents a semantic column grouping system designed to identify and standardize attribute naming conventions in CSV datasets. The system leverages a locally deployed Large Language Model (LLaMA3 via Ollama) to perform semantic analysis, enabling accurate grouping of columns that represent the same concept despite variations in naming.

Traditional string-based methods often fail to capture semantic relationships between attributes such as "fname", "first_name", and "givenName". This system overcomes such limitations by using contextual understanding through LLMs.

---

## Features

- Upload CSV files via a web interface
- Automatic column extraction and normalization
- Semantic grouping using a locally deployed LLM
- Robust handling of abbreviations, typos, and case variations
- JSON-based structured output
- Export grouped results as CSV
- Fully local execution (privacy-preserving)

---

## Technologies Used

- Python
- Streamlit
- Ollama
- LLaMA3
- Pandas
- Regular Expressions (Regex)

## How to Run

```bash
pip install -r requirements.txt
ollama serve
streamlit run app.py
```

