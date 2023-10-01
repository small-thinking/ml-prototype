import argparse
import os

import torch
from tokenizer import BytePairTokenizer, NaiveTokenizer, Tokenizer
from torch.nn.functional import softmax


class InferenceEngine:
    def __init__(
        self,
        tokenizer_type: str,
        token_folder_path: str,
        jit_model_path: str,
        device: str,
    ):
        self.device = device
        self.model = torch.jit.load(jit_model_path).to(self.device)
        self.model.eval()
        self.tokenizer_type = tokenizer_type
        if self.tokenizer_type == "bpe":
            self.tokenizer = BytePairTokenizer(
                config={"token_folder_path": token_folder_path}
            )
        else:
            self.tokenizer = NaiveTokenizer(
                config={"token_folder_path": token_folder_path}
            )

    def inference(self, text: str, max_length: int = 256, temperature: float = 1.0):
        print(f"Inference: {text}\n")
        input_tensor = self.tokenizer.encode(text).unsqueeze(0).to(self.device)
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

                next_token = torch.multinomial(adj_prob, num_samples=1).to(self.device)
                generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

                # Decode the next token to text and yield it
                next_token_text = self.tokenizer.decode(next_token[0])[0]
                yield next_token_text

                if generated_sequence.size(1) >= max_length:
                    break


def main():
    # Initialize argparse object
    parser = argparse.ArgumentParser(description="Inference Script")

    # Add arguments
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="To be or not to be. This is a question. What is the matter of the universe?",
        help="Prompt.",
    )
    parser.add_argument(
        "--max_length",
        "-m",
        type=int,
        default=256,
        help="Maximum length of generated sequence.",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=1.0, help="Temperature for sampling."
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda:0", help="The device to use."
    )
    parser.add_argument(
        "--tokenizer_type",
        "-tt",
        type=str,
        default="char",
        help='Tokenizer type: "char" or "bpe".',
    )
    parser.add_argument(
        "--token_folder_path",
        "-tf",
        type=str,
        default="./tokenizer/char",
        help="The folder path of the token.",
    )

    # Parse the arguments
    args = parser.parse_args()
    inference_engine = InferenceEngine(
        tokenizer_type=args.tokenizer_type,
        token_folder_path=args.token_folder_path,
        jit_model_path=os.path.join(
            os.path.dirname(__file__), "../../model_epoch_30.pt"
        ),
        device=args.device,
    )

    # Use parsed arguments
    for next_token_text in inference_engine.inference(
        args.prompt, args.max_length, args.temperature
    ):
        print(next_token_text, end="")


if __name__ == "__main__":
    main()
