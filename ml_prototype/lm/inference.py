import os

import torch
from tokenizer import NaiveTokenizer, Tokenizer
from torch.nn.functional import softmax

device = "cpu"
device = "cuda:0"


class InferenceEngine:
    def __init__(self, tokenizer, jit_model_path: str):
        self.model = torch.jit.load(jit_model_path).to(device)  # Move model to CPU
        self.model.eval()
        self.tokenizer = tokenizer

    def inference(self, text: str, max_length: int = 128, temperature: float = 1.0):
        print(f"Inference: {text}")
        input_tensor = self.tokenizer.encode([text]).to(device)  # Move tensor to CPU
        generated_sequence = input_tensor

        with torch.no_grad():
            for _ in range(max_length):
                generated_sequence_crop = generated_sequence[:, -256:]
                logits = self.model(generated_sequence_crop)
                # Get the logit of the last token
                last_logits = logits[:, -1, :]
                # Adjust logits with temperature
                adj_logits = last_logits / temperature
                # Apply softmax to get probabilities
                adj_prob = softmax(adj_logits, dim=-1)

                next_token = torch.multinomial(adj_prob, num_samples=1).to(
                    device
                )  # Move tensor to CPU
                generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

                # Decode the next token to text and yield it
                next_token_text = self.tokenizer.decode(next_token[0])
                yield next_token_text

                if generated_sequence.size(1) >= max_length:
                    break


def main():
    token_file = os.path.expanduser("./data/tokens.json")
    tokenizer = NaiveTokenizer(config={}, token_file=token_file)
    inference_engine = InferenceEngine(
        tokenizer=tokenizer,
        jit_model_path=os.path.join(
            os.path.dirname(__file__), "../../model_epoch_20.pt"
        ),
    )

    text = "To be or not to be. This is a question. What is the matter of the universe?"
    for next_token_text in inference_engine.inference(text[:256]):
        print(next_token_text, end="")


if __name__ == "__main__":
    main()
