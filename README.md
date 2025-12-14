# Sarcasm Detection in News Headlines

**Author:** Deogyong Kim (2021134015)  
**Course:** CAS2105 - Mini AI Pipeline Project

---

## Overview

This project compares five approaches for detecting sarcasm in news headlines:
1. Naive Baseline (rule-based)
2. Embedding + Centroid
3. Embedding + Logistic Regression
4. LLM Zero-shot
5. LLM Few-shot

**Key Finding:** Embedding + LR achieved the best performance (84.71% accuracy, F1=0.84) while being 227× faster than LLM methods.

---

## Requirements

### Environment
- Python 3.10
- CUDA-capable GPU (RTX 3090 24GB VRAM)
- Ubuntu 24.04 or similar Linux environment

### Installation

The notebook includes automatic package installation in the first cell:
```python
!pip install pandas numpy scikit-learn matplotlib seaborn tqdm torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 transformers==4.51.3 sentence-transformers accelerate autoawq openai huggingface_hub
```

---

## Dataset

The **Sarcasm Headlines Dataset v2** is automatically downloaded within the notebook.
- Source: [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- No manual download required - the notebook handles data loading

---

## Usage

### Run Jupyter Notebook
```bash
jupyter notebook pipeline_demo.ipynb
```

Run all cells sequentially. The notebook includes:
- **Automatic package installation**
- **Automatic dataset download**
- **Automatic model download**
- Data exploration
- All five methods implementation
- Performance comparison
- Case analysis
- Timing measurements

---

## Repository Structure
```
sarcasm-detection/
├── README.md                              # This file
├── report.pdf                             # Full project report
├── data/
│   └── Sarcasm_Headlines_Dataset_v2.json  # Downloaded by notebook
├── model/
│   └── Qwen2.5-32B-Instruct-AWQ/          # Downloaded by notebook
└── notebook/
    └── pipeline_demo.ipynb                # Main notebook (all-in-one)
```

**Note:** 
- Dataset and Model are downloaded during notebook execution
---

## Key Results

| Method | Accuracy | F1 Score | Time per Sample |
|--------|----------|----------|-----------------|
| Naive Baseline | 56.46% | 0.24 | 0.03 ms |
| Embedding + Centroid | 74.72% | 0.73 | 10.38 ms |
| **Embedding + LR** | **84.71%** | **0.84** | **8.88 ms** |
| LLM Zero-shot | 81.87% | 0.80 | 1398.77 ms |
| LLM Few-shot | 82.95% | 0.84 | 2017.22 ms |

**Best Method:** Embedding + LR achieves highest accuracy with excellent efficiency.

---

## Notes

- **GPU Memory:** LLM (Qwen2.5-32B-Instruct-AWQ) requires ~18GB VRAM and fits on single RTX 3090
- **Reproducibility:** All experiments use `random_seed=42` for reproducibility
- **Model Path:** Default model path is `~/model/Qwen2.5-32B-Instruct-AWQ`. Update in notebook if needed
- **Execution Time:** Full pipeline takes ~4-5 hours (mostly LLM inference)
