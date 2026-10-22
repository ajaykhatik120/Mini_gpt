# Mini GPT From Scratch

A small, educational GPT-style character-level language model implemented in **Python** and **PyTorch**. The project demonstrates the core stages of a language-modeling pipeline: building a character vocabulary, converting text to token IDs, learning next-character patterns with causal self-attention, and generating text one character at a time.

> **Scope:** This is a learning implementation, not a production chatbot or a reproduction of a large GPT model. It is trained from scratch on the small text file included in this repository and generates short continuations from that limited training data.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the Project](#run-the-project)
- [Configuration](#configuration)
- [Training and Generation Details](#training-and-generation-details)
- [Tests](#tests)
- [Current Limitations](#current-limitations)
- [Ideas for Improvements](#ideas-for-improvements)

## Overview

Mini GPT is designed to make the mechanics of a very small autoregressive language model easier to inspect.

The implementation includes:

- **Character-level tokenization:** each distinct character in the training text is assigned an integer ID.
- **Token and positional embeddings:** character IDs and their sequence positions are mapped to learned vectors.
- **Causal multi-head self-attention:** the model uses PyTorch's multi-head attention with a causal mask so a position cannot attend to future positions.
- **Next-character training:** the model learns to predict the next character from preceding characters using cross-entropy loss and the Adam optimizer.
- **Autoregressive generation:** the model repeatedly predicts the next character and appends it to the sequence.
- **Command-line interaction:** an interactive loop lets you enter prompts and see short continuations.

## How It Works

The program follows this pipeline:

1. **Load text.** Read `data/story.txt` as the training corpus.
2. **Build a vocabulary.** `CharTokenizer` collects the unique characters and creates character-to-ID and ID-to-character mappings.
3. **Encode the corpus.** The text becomes a sequence of integer token IDs.
4. **Initialize the model.** `SimpleGPT` creates token embeddings, positional embeddings, multi-head attention, and an output projection to vocabulary logits.
5. **Train next-character prediction.** For each training chunk, the input is the chunk except its final character; the target is the same chunk shifted by one character.
6. **Generate a continuation.** Starting from a prompt, the model predicts the most likely next character, appends it, and repeats for the configured number of new characters.

Conceptually, for a sequence such as `Once`, the training pairs are shifted:

- Input: `O n c`
- Target: `n c e`

The model is optimized to assign higher probability to the correct next character at each position.

## Architecture

### Model: `model/gpt_model.py`

The `SimpleGPT` module contains:

- **Token embedding:** maps each character ID to a learned vector.
- **Position embedding:** adds a learned representation of each position in the current context.
- **Multi-head attention:** uses `torch.nn.MultiheadAttention` with four attention heads by default.
- **Causal mask:** applies a subsequent-position mask to prevent attention to future tokens during training and generation.
- **Output projection:** maps each contextualized vector to one score (logit) per vocabulary character.

The default model settings are intentionally small: an embedding width of 32, four attention heads, and a context length of 32 characters.

### Tokenizer: `tokenizer/char_tokenizer.py`

`CharTokenizer` builds a sorted vocabulary from the supplied text. It exposes:

- `encode(text)`: returns a list of character IDs.
- `vocab_size`: returns the number of characters in the vocabulary.
- `stoi`: character-to-index mapping.
- `itos`: index-to-character mapping.

This tokenizer does not split text into words or subword pieces. Any character not present in the training corpus is outside the vocabulary.

### Trainer: `training/trainer.py`

The trainer:

- creates an Adam optimizer;
- splits the encoded corpus into chunks based on the model's context length;
- creates shifted input and target tensors;
- computes cross-entropy loss;
- backpropagates gradients and updates the parameters.

### Text generation: `inference/generate.py`

Generation runs the model in evaluation mode and disables gradient tracking. At each step it takes the last time-step's logits and chooses the highest-scoring token with `argmax`. That token is appended to the generated sequence.

This is **greedy decoding**: it is deterministic for a fixed model and prompt and does not sample from a probability distribution.

## Repository Structure

```text
Mini_gpt/
├── README.md
└── mini_gpt_from_scratch/
    ├── main.py                   # Train the model and generate a sample continuation
    ├── chat.py                   # Train the model, then start an interactive prompt loop
    ├── requirements.txt          # Python dependencies
    ├── data/
    │   └── story.txt             # Small example training corpus
    ├── model/
    │   └── gpt_model.py          # Embeddings, causal attention, output projection
    ├── tokenizer/
    │   └── char_tokenizer.py     # Character vocabulary and ID mappings
    ├── training/
    │   └── trainer.py            # Training loop and next-character loss
    ├── inference/
    │   └── generate.py           # Greedy autoregressive text generation
    ├── utils/
    │   └── config.py             # Default hyperparameters
    ├── tests/
    │   └── test_training.py      # Basic training smoke test
    └── visualization/             # Reserved for future visualization work
```

Python cache folders such as `__pycache__` may appear after running the code and are not part of the model architecture.

## Requirements

- Python 3.10+ recommended
- PyTorch 2.0 or later

The dependency file currently contains only:

```text
torch>=2.0.0
```

The code is CPU-compatible; a GPU is not required for this tiny example.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/ajaykhatik120/Mini_gpt.git
cd Mini_gpt/mini_gpt_from_scratch
```

Create and activate a virtual environment (recommended):

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Project

Run a single training-and-generation example:

```bash
python main.py
```

This loads the included story, fits the tokenizer, trains the model, prints the final training loss, and generates a continuation from the built-in prompt `Once upon a`.

To start an interactive prompt session:

```bash
python chat.py
```

The chat script **trains a fresh model every time it starts**. After training, enter a prompt and press Enter. Type `quit`, `exit`, or `q` to leave. Prompts must use only characters found in the training corpus; the script checks for unknown characters before generation.

Example interaction (actual output will vary):

```text
You > Once upon a
GPT > Once upon a ...
```

The ellipsis is illustrative, not a guaranteed literal output. Since the included corpus is extremely small, generated text may be repetitive or incoherent.

## Configuration

Defaults are defined in `utils/config.py`:

| Parameter | Default | Meaning |
|---|---:|---|
| `block_size` | 32 | Maximum context length in characters |
| `epochs` | 50 | Number of training epochs |
| `max_new_tokens` | 30 | Number of new characters generated |

The model constructor also defaults to `n_embd=32` and `n_head=4`. The trainer's default learning rate is `1e-3`.

To experiment, edit the `Config` dataclass or pass different values when constructing the model and calling the trainer. The context length must be compatible with the model's positional embedding size.

## Training and Generation Details

- **Objective:** next-character language modeling.
- **Loss:** cross-entropy over the vocabulary at each position.
- **Optimizer:** Adam.
- **Attention:** causal multi-head self-attention.
- **Decoding:** greedy selection using `argmax`.
- **Training corpus:** `data/story.txt`, a short, hand-written sample text.
- **Checkpointing:** the current scripts do not save or load trained model weights; the model is retrained on each run.

Because the corpus is only a few sentences long, this project is best used to inspect model mechanics and experiment with code rather than to evaluate language quality.

## Tests

From the `mini_gpt_from_scratch` directory, run:

```bash
python -m unittest discover -s tests -v
```

The current test is a smoke test that checks the training function runs on a short sample and returns a non-negative loss. It does not measure generation quality, generalization, or model correctness comprehensively.

## Current Limitations

- **Very small dataset:** the included text is only a few sentences, so the model has little linguistic variety to learn from.
- **Character-level vocabulary:** the model predicts one character at a time rather than words or subword tokens.
- **Small architecture:** this is a minimal GPT-style model, not a full-scale GPT implementation with stacked Transformer blocks, feed-forward sublayers, layer normalization, residual connections, or dropout.
- **Simple batching and optimization:** training uses one chunk at a time and reports the average loss of the final epoch's chunks rather than a full training history.
- **Greedy decoding:** always choosing the most likely next character can create repetitive output.
- **No checkpoint support:** parameters are not persisted after a run.
- **No unknown-token ID:** prompts containing unseen characters cannot be encoded; the interactive script detects this, but direct calls to `CharTokenizer.encode` can raise a key error.
- **No visualization implementation yet:** the `visualization/` directory currently contains only package scaffolding.

## Ideas for Improvements

Potential next steps include:

1. Add a validation split and track training and validation loss across epochs.
2. Save and load model checkpoints so training does not need to run for every session.
3. Add temperature, top-k, or top-p sampling to provide alternatives to greedy decoding.
4. Improve the training loop with randomized batches and more systematic evaluation.
5. Add input validation and an explicit unknown-character policy in the tokenizer.
6. Experiment with a larger and more diverse training corpus.
7. Extend the architecture with stacked Transformer blocks, feed-forward layers, normalization, residual connections, and dropout.
8. Implement plotting utilities for training and validation metrics.

## License

No license file was identified in the repository when this README was prepared. Add a license file if you intend to define how others may use, modify, and distribute the project.
