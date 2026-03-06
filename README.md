
# 🧠 RNN-based IMDB Sentiment Analysis (PyTorch)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![NLP](https://img.shields.io/badge/Task-Sentiment%20Analysis-green)

This repository implements an **end‑to‑end Recurrent Neural Network (RNN) pipeline** for **binary sentiment classification** on the **IMDB movie review dataset** using **PyTorch**.

The project demonstrates the full machine learning workflow including:

- Text preprocessing and tokenization
- Vocabulary construction
- Sequence padding and encoding
- Training recurrent neural architectures
- Hyperparameter exploration
- Model evaluation using **accuracy, loss curves, and ROC/AUC**
- Inference on unseen reviews

---

# 🚀 Project Highlights

✔ Full NLP preprocessing pipeline  
✔ PyTorch implementation of **Embedding + RNN architecture**  
✔ Comparison of **SimpleRNN, LSTM, and GRU models**  
✔ Hyperparameter tuning and analysis  
✔ Evaluation with **Accuracy and ROC‑AUC**  

---

# 🏗 Model Architecture

Input Text  
→ Tokenization  
→ Padding  
→ Embedding Layer  
→ Recurrent Layer (GRU / LSTM / SimpleRNN)  
→ Dropout  
→ Fully Connected Layer  
→ Sigmoid Output  

---

# 📊 Experimental Results

## Baseline Model

| Metric | Result |
|---|---|
| Training Accuracy | **98.03%** |
| Validation Accuracy | **84.21%** |
| ROC‑AUC | **0.8415** |

---

# 🔬 Hyperparameter Exploration

Three RNN variants were evaluated:

- **LSTM**
- **SimpleRNN**
- **GRU**

Key findings:

- **GRU achieved the most stable performance**
- Learning rate **0.001 produced the best training stability**
- Increasing hidden layers beyond **3 reduced performance**
- Larger batch sizes improved computational efficiency

---

# 🏆 Best Model Configuration

| Parameter | Value |
|---|---|
| Model | GRU |
| Learning Rate | 0.001 |
| Hidden Layers | 2 |
| Batch Size | 75 |
| Embedding Dimension | 64 |
| Hidden Dimension | 256 |
| Sequence Length | 500 |

### Final Model Performance

| Metric | Result |
|---|---|
| Training Accuracy | **99.07%** |
| Validation Accuracy | **85.21%** |
| ROC‑AUC | **0.8521** |

---

# 📂 Repository Structure

.
├── RNN_IMDB_Sentiment_Analysis.ipynb  
├── README.md  
└── dataset/

---

# ⚙️ Installation

Clone the repository:

git clone https://github.com/mhafezan/RNN_IMDB_Sentiment_Analysis.git

Install dependencies:

pip install numpy pandas torch nltk scikit-learn matplotlib seaborn tqdm

---

# ▶️ Running the Project

Open the notebook:

RNN_IMDB_Sentiment_Analysis.ipynb

Run all cells sequentially to:

1. Load dataset
2. Preprocess text
3. Train the RNN model
4. Evaluate performance
5. Visualize results

---

# 🔮 Future Improvements

- Use pretrained embeddings (GloVe / Word2Vec)
- Explore Transformer models such as **BERT**
- Implement CNN‑RNN hybrid models
- Deploy model as an inference API

---

# 👨‍💻 Author

**Mohammad Hafezan**
