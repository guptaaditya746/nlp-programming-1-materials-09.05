# BPE Tokenizer and SkipGram Model Workspace

This workspace contains Python scripts for training a Byte Pair Encoding (BPE) tokenizer and a SkipGram word embedding model.

## Project Structure

```
.
├── bpe.py             # BPE Tokenizer implementation, training, and testing
├── skipgram.py        # SkipGram model implementation, training, and testing
├── imdb.txt           # Dataset file (needs to be present)
├── output/            # Directory for saved models and test outputs (created by scripts)
└── README.md          # This file
```

## 1. Setup

### Prerequisites

*   Python 3.8 or higher is recommended.
*   `pip` for installing Python packages.

### Dependencies

The project relies on the following Python libraries:
*   `torch` (for PyTorch, used in SkipGram)
*   `tqdm` (for progress bars)
*   `matplotlib` (for plotting training loss)
*   `pytest` (for running tests)

You can install these dependencies using pip. It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch tqdm matplotlib pytest
```

Alternatively, you can create a `requirements.txt` file with the following content:

```txt
torch
tqdm
matplotlib
pytest
```

And then install using:

```bash
pip install -r requirements.txt
```

### Dataset

The `imdb.txt` file is required for training. Ensure it is placed in the root directory of this workspace (`/home/prims/Downloads/nlp-programming-1-materials-09.05/`).

## 2. Configuration

Before running the scripts, you need to set your group number in both `bpe.py` and `skipgram.py`.

Open `/home/prims/Downloads/nlp-programming-1-materials-09.05/bpe.py` and `/home/prims/Downloads/nlp-programming-1-materials-09.05/skipgram.py` and modify the `GROUP` variable at the top of each file:

```python
GROUP = "XX"  # Replace XX with your actual group number
```

## 3. Running the Code

### Training the BPE Tokenizer

To train the BPE tokenizer and save the model:

```bash
python /home/prims/Downloads/nlp-programming-1-materials-09.05/bpe.py
```

This will:
*   Load the `imdb.txt` dataset.
*   Train the BPE tokenizer based on the parameters in the `main` function of `bpe.py`.
*   Save the trained tokenizer to `./output/bpe_group_XX.json` (where XX is your group number).
*   Generate a sample tokenization output to `./output/bpe_test_group-XX.txt`.

### Training the SkipGram Model

After the BPE tokenizer is trained and saved, you can train the SkipGram model:

```bash
python /home/prims/Downloads/nlp-programming-1-materials-09.05/skipgram.py
```

This will:
*   Load the previously saved BPE tokenizer.
*   Load the `imdb.txt` dataset.
*   Train the SkipGram model.
*   Save the trained model to `./output/skipgram_group_XX.pt`.
*   Generate a `training_loss.png` plot.
*   Output similar tokens for test samples to `./output/skipgram_test_group_XX.txt`.

## 4. Running Tests

To run the automated tests for each module:

```bash
pytest /home/prims/Downloads/nlp-programming-1-materials-09.05/bpe.py
pytest /home/prims/Downloads/nlp-programming-1-materials-09.05/skipgram.py
```

Ensure all tests pass to verify the correctness of your implementations.

## 5. Output Files

The scripts will generate the following important files:

*   `/home/prims/Downloads/nlp-programming-1-materials-09.05/output/bpe_group_XX.json`: Saved BPE tokenizer model.
*   `/home/prims/Downloads/nlp-programming-1-materials-09.05/output/skipgram_group_XX.pt`: Saved SkipGram model.
*   `/home/prims/Downloads/nlp-programming-1-materials-09.05/output/bpe_test_group-XX.txt`: Example tokenizations from BPE.
*   `/home/prims/Downloads/nlp-programming-1-materials-09.05/output/skipgram_test_group_XX.txt`: Example similar words from SkipGram.
*   `/home/prims/Downloads/nlp-programming-1-materials-09.05/training_loss.png`: Plot of the SkipGram training loss.

*(Replace XX with your actual group number in the filenames above)*