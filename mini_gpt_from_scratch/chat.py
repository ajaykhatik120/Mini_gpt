"""Interactive prompt loop for the mini GPT model.

Usage:
    python chat.py

The script trains the model on data/story.txt, then drops you into
an interactive session where you can type any starting text and the
model will attempt to continue it.

Type 'quit' or 'exit' to leave.  Press Ctrl+C at any time to stop.
"""

from pathlib import Path

from inference.generate import generate_text
from model.gpt_model import SimpleGPT
from tokenizer.char_tokenizer import CharTokenizer
from training.trainer import train_simple_model
from utils.config import Config


def main() -> None:
    config = Config()

    # ── Load and tokenize training data ──────────────────────────
    data_path = Path(__file__).resolve().parent / "data" / "story.txt"
    story = data_path.read_text(encoding="utf-8")

    tokenizer = CharTokenizer()
    tokens = tokenizer.encode(story)

    # ── Train ────────────────────────────────────────────────────
    model = SimpleGPT(vocab_size=tokenizer.vocab_size, block_size=config.block_size)
    print("Training the model …")
    loss = train_simple_model(model, tokens, epochs=config.epochs)
    print(f"Training complete  (final loss: {loss:.4f})\n")

    # ── Interactive loop ─────────────────────────────────────────
    print("=" * 50)
    print("  Mini GPT – Interactive Text Completion")
    print("=" * 50)
    print(f"  Vocabulary : {tokenizer.vocab_size} characters")
    print(f"  Block size : {config.block_size}")
    print(f"  New tokens : {config.max_new_tokens} per completion")
    print("-" * 50)
    print("Type a prompt and press Enter to see the model")
    print("continue your text.  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("You > ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.strip().lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not prompt:
            continue

        # Check that every character in the prompt is in the vocabulary
        unknown = [ch for ch in prompt if ch not in tokenizer.stoi]
        if unknown:
            print(f"  ⚠  Unknown character(s): {set(unknown)}")
            print(f"      Known vocabulary: {''.join(tokenizer.vocab)}\n")
            continue

        generated = generate_text(
            model, tokenizer, prompt, max_new_tokens=config.max_new_tokens
        )
        print(f"GPT > {generated}\n")


if __name__ == "__main__":
    main()
