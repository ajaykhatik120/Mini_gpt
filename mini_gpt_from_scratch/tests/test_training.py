import unittest

from tokenizer.char_tokenizer import CharTokenizer
from model.gpt_model import SimpleGPT
from training.trainer import train_simple_model


class TestTraining(unittest.TestCase):
    def test_train_simple_model_runs_for_longer_text(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz"
        tokenizer = CharTokenizer()
        tokens = tokenizer.encode(text)
        model = SimpleGPT(vocab_size=tokenizer.vocab_size, block_size=32)

        loss = train_simple_model(model, tokens, epochs=1)

        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
