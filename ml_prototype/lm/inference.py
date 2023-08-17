import os

import torch
from torch.nn.functional import softmax

from ml_prototype.lm.module import LanguageModule, TransformerLM
from ml_prototype.lm.tokenizer import NaiveTokenizer, Tokenizer


class InferenceEngine:
    def __init__(self, tokenizer: Tokenizer, jit_model_path: str):
        # Load the model checkpoint
        self.model = torch.jit.load(jit_model_path)
        self.model.eval()  # Set the model to evaluation mode
        self.tokenizer = tokenizer

    def inference(self, text: str, max_length: int = 50):
        # Tokenize the input text
        input_tensor = self.tokenizer.encode([text])

        # Generate predictions
        with torch.no_grad():
            logits = self.model(input_tensor)

        # Apply a softmax to get probabilities
        probabilities = softmax(logits, dim=-1)

        # Decode the output tensor to text
        decoded_text = self.tokenizer.decode(logits.argmax(dim=-1))

        return decoded_text, probabilities


def main():
    doc_file_path = os.path.expanduser("~/Downloads/test_data.txt")
    tokenizer = NaiveTokenizer(config={}, doc_file_path=doc_file_path)
    inference_engine = InferenceEngine(tokenizer=tokenizer, jit_model_path="model.pt")

    text = "The quick brown fox"
    decoded_text, probabilities = inference_engine.inference(text[:16])

    print("Predicted Text:", decoded_text)


if __name__ == "__main__":
    main()
