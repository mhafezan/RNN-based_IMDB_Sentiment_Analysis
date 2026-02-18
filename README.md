# RNN-based IMDB Sentiment Analysis (PyTorch)

This repository contains a single Jupyter notebook that builds an end-to-end **binary sentiment classifier** for the **IMDB movie reviews** dataset. The workflow covers dataset loading, text preprocessing (tokenization, stop-word filtering, vocabulary building, sequence padding), model training with an **Embedding + RNN backbone (GRU/LSTM/SimpleRNN-ready)**, and evaluation using accuracy, loss curves, and ROC/AUC.  

> Notebook: `RNN_IMDB_Sentiment_Analysis.ipynb`

---

## What this notebook does

- Loads the IMDB CSV dataset (`review`, `sentiment`)
- Splits data into train/test sets (80/20)
- Cleans and tokenizes text, removes stop-words, builds a frequency-sorted vocabulary
- Encodes labels (positive→1, negative→0)
- Pads/truncates sequences to a fixed length (default: **500** tokens)
- Trains a recurrent model (default configuration uses **GRU**) for sentiment prediction
- Visualizes training/validation accuracy and loss
- Computes ROC curve and AUC
- Runs inference on sample reviews

---

## Requirements

Install the core dependencies:

```bash
pip install numpy pandas torch nltk scikit-learn matplotlib tqdm seaborn
```

NLTK stopwords are downloaded in the notebook via:
```python
nltk.download("stopwords")
```

---

## Dataset

The notebook expects the **IMDB Dataset** CSV (commonly named `IMDB_Dataset.csv`) with at least:
- `review`: raw text review
- `sentiment`: `positive` or `negative`

Update the dataset path in **Cell 3** to match your environment (local path, Colab drive, etc.).

---

## Notebook walkthrough (cell-by-cell)

### Cell 1 — Imports & NLP utilities
- Imports NumPy/Pandas, PyTorch modules, Scikit-learn helpers, Matplotlib/Seaborn
- Downloads NLTK stopwords and constructs the `stop_words` set

### Cell 2 — Device initialization (CPU/GPU)
- Selects a compute device for training and inference  
**Note:** use `torch.cuda.is_available()` (with parentheses) for correct GPU detection.

### Cell 3 — Load and preview dataset
- Loads the IMDB CSV into a Pandas DataFrame
- Displays sample rows to verify schema and content

### Cell 4 — Train/test split
- Extracts `x = df["review"]` and `y = df["sentiment"]`
- Splits into train/test using `train_test_split(test_size=0.20)`

### Cell 5 — Text cleanup helper: `preprocess_string`
- Removes punctuation, whitespace runs, and digits using regex
- Produces normalized tokens used throughout tokenization

### Cell 6 — Sequence padding helper: `padding`
- Converts variable-length token lists into a fixed-size NumPy matrix
- Right-aligns sequences and truncates longer sequences to `seq_len`

### Cell 7 — Tokenization + vocabulary builder: `tockenize`
- Builds a word corpus from training text (after cleanup + stop-word filtering)
- Creates a frequency-sorted vocabulary mapping (`one_hot_dict`)
- Converts reviews to integer sequences using the vocabulary
- Encodes labels to 0/1
- Pads sequences to `max_length = 500`
- Returns padded train/test arrays and the vocabulary

### Cell 8 — Apply tokenization pipeline
- Runs `tockenize(...)` on train/test split
- Prints vocabulary size and confirms ordering by word frequency

### Cell 9 — PyTorch DataLoaders
- Wraps padded arrays into `TensorDataset`
- Creates `DataLoader` objects for training and validation
- Prints loader sizes and sample tensor shapes

### Cell 10 — Model definition: `SentimentRNN`
- Defines a configurable RNN classifier:
  - `nn.Embedding` for dense word vectors
  - RNN backbone (GRU used by default; code is structured to support LSTM or SimpleRNN)
  - Dropout regularization
  - Fully-connected layer + Sigmoid for binary probability output
- Includes `init_hidden(...)` compatible with GRU/SimpleRNN (and an LSTM alternative in comments)

### Cell 11 — Hyperparameters, loss, optimizer, accuracy helper
- Sets model hyperparameters (layers, embedding dim, hidden dim, dropout)
- Instantiates the model on the selected device
- Uses `BCELoss` + Adam optimizer
- Defines an `acc(...)` helper for batch accuracy

### Cell 12 — Training + validation loop
- Trains for `epochs = 10`
- Uses gradient clipping (`clip_value = 5`) to reduce exploding gradients
- Tracks per-epoch train/validation loss and accuracy
- Prints epoch summaries
- Notes a “saving model” message when validation loss improves (you can add `torch.save(...)` to persist checkpoints)

### Cell 13 — Learning curves (Accuracy & Loss)
- Plots training vs. validation accuracy and loss over epochs

### Cell 14 — ROC curve and AUC
- Collects predictions on the validation set
- Computes ROC curve and AUC, then plots ROC
**Note:** AUC/ROC is typically computed using probabilities (not thresholded labels). Consider using raw sigmoid outputs for a smoother ROC.

### Cell 15 — Inference on reviews
- Implements `predict_text(text)` for single-review prediction
- Demonstrates predictions on selected dataset rows and prints:
  - Review text
  - Actual sentiment
  - Predicted sentiment + confidence

---

## How to run

### Option 1: Jupyter / VS Code
1. Open `RNN_IMDB_Sentiment_Analysis.ipynb`
2. Set the dataset CSV path in **Cell 3**
3. Run cells from top to bottom

### Option 2: Google Colab
1. Upload the notebook to Colab
2. Mount Google Drive (if using Drive paths)
3. Install dependencies if needed
4. Run top-to-bottom

---

## Notes / suggested improvements (optional)

- **Fix GPU detection:** `torch.cuda.is_available()` in Cell 2  
- **Persist the best checkpoint:** add `torch.save(model.state_dict(), "model.pt")` when validation improves  
- **Use probabilities for ROC/AUC:** compute ROC using sigmoid probabilities instead of thresholded outputs  
- **Reproducibility:** set random seeds for NumPy/PyTorch and pass `random_state` to `train_test_split`
