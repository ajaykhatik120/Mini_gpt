from pathlib import Path

from inference.generate import generate_text
from model.gpt_model import SimpleGPT
from tokenizer.char_tokenizer import CharTokenizer
from training.trainer import train_simple_model
from utils.config import Config


def main() -> None:
    config = Config()
    data_path = Path(__file__).resolve().parent / "data" / "story.txt"
    story = data_path.read_text(encoding="utf-8")

    tokenizer = CharTokenizer()
    tokens = tokenizer.encode(story)

    model = SimpleGPT(vocab_size=tokenizer.vocab_size, block_size=config.block_size)
    loss = train_simple_model(model, tokens, epochs=config.epochs)

    prompt = "Once upon a"
    generated = generate_text(model, tokenizer, prompt, max_new_tokens=config.max_new_tokens)

    print(f"Training loss: {loss:.4f}")
    print("Generated text:")
    print(generated)


if __name__ == "__main__":
    main()
