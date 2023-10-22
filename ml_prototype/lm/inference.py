import argparse
import os

import torch
from tokenizer import (
    BytePairTokenizer,
    NaiveTokenizer,
    SentencePieceTokenizer,
    Tokenizer,
)
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
        elif self.tokenizer_type == "spm":
            self.tokenizer = SentencePieceTokenizer(
                config={"token_folder_path": token_folder_path}
            )
        else:
            self.tokenizer = NaiveTokenizer(
                config={"token_folder_path": token_folder_path}
            )

    def inference(
        self,
        text: str,
        max_length: int = 256,
        temperature: float = 1.0,
        seq_len: int = 64,
    ):
        print(f"Inference: {text}\n")
        input_tensor = self.tokenizer.encode(text).unsqueeze(0).to(self.device)
        generated_sequence = input_tensor

        with torch.no_grad():
            for _ in range(max_length):
                generated_sequence_crop = generated_sequence[:, -seq_len:]
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
                try:
                    next_token_text = self.tokenizer.decode(next_token[0])[0]
                    yield next_token_text
                except Exception:
                    print(f"Error next token: '{self.tokenizer.decode(next_token[0])}'")
                    print(f"Next token: {next_token[0]}")
                    exit()

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
        "--seq_len",
        "-s",
        type=int,
        default=512,
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
        default="bpe",
        help='Tokenizer type: "char", "spm" or "bpe".',
    )
    parser.add_argument(
        "--token_folder_path",
        "-tf",
        type=str,
        default="./tokenizer/bpe-256",
        help="The folder path of the token.",
    )

    # Parse the arguments
    args = parser.parse_args()
    print(args)
    inference_engine = InferenceEngine(
        tokenizer_type=args.tokenizer_type,
        token_folder_path=args.token_folder_path,
        jit_model_path=os.path.join(
            os.path.dirname(__file__), "../../model_epoch_10.pt"
        ),
        device=args.device,
    )

    # Use parsed arguments
    for next_token_text in inference_engine.inference(
        text=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        seq_len=args.seq_len,
    ):
        print(next_token_text, end="")


if __name__ == "__main__":
    main()
