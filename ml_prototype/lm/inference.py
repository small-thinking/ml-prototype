import os

import torch
from tokenizer import NaiveTokenizer, Tokenizer
from torch.nn.functional import softmax


class InferenceEngine:
    def __init__(self, tokenizer: Tokenizer, jit_model_path: str):
        # Load the model checkpoint
        self.model = torch.jit.load(jit_model_path)
        self.model.eval()  # Set the model to evaluation mode
        self.tokenizer = tokenizer

    def inference(self, text: str, max_length: int = 50):
        print(f"Inference: {text}")
        # Tokenize the input text
        input_tensor = self.tokenizer.encode([text])
        # Initialize generated sequence with the input tensor
        generated_sequence = input_tensor

        # Generate predictions in an autoregressive way
        with torch.no_grad():
            for _ in range(max_length):
                generated_sequence_crop = generated_sequence[:, -127:]
                logits = self.model(generated_sequence_crop)
                # Take the logits corresponding to the last token
                last_logits = logits[:, -1, :]
                # Apply a softmax to get probabilities
                probabilities = softmax(last_logits, dim=-1)
                # Get the most likely token
                next_token = torch.argmax(probabilities, dim=-1).unsqueeze(1)

                # Check whether the generated sequence is less than context_size
                if generated_sequence.size(1) < max_length:
                    # Append the next token without removing any tokens
                    generated_sequence = torch.cat(
                        [generated_sequence, next_token], dim=1
                    )
                else:
                    # Remove the first token and append the next token
                    generated_sequence = generated_sequence[:, 1:]
                    generated_sequence = torch.cat(
                        [generated_sequence, next_token], dim=1
                    )

        # Decode the generated sequence to text
        decoded_text = self.tokenizer.decode(generated_sequence[0])

        return decoded_text


def main():
    doc_file_path = os.path.expanduser("~/Downloads/tiny-shakespeare.txt")
    tokenizer = NaiveTokenizer(config={}, doc_file_path=doc_file_path)
    inference_engine = InferenceEngine(
        tokenizer=tokenizer,
        jit_model_path=os.path.join(os.path.dirname(__file__), "../../model.pt"),
    )

    text = "This is hello world. This is our"
    decoded_text = inference_engine.inference(text[:128])

    print(f"Predicted Text: [{decoded_text}]")


if __name__ == "__main__":
    main()
