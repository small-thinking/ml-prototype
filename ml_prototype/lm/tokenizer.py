import abc
import argparse
import json
from glob import glob
import os
from typing import Any, Dict, List, Sequence, Union

from tokenizers import ByteLevelBPETokenizer 
import torch


class Tokenizer(abc.ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def vocab_size(self):
        """Get the cardinality of the vocab."""
        raise NotImplementedError

    def train(self):
        """Train the tokenizer. Not all tokenizer are trainable.
        """
        pass

    @abc.abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """Encode the input texts.

        Args:
            text (str): The input text.

        Returns:
            torch.Tensor: The tokenized texts in shape of [seq_len].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, tensor: torch.Tensor) -> List[str]:
        """Decode the tensor back to text.

        Args:
            tensor (torch.Tensor): The tokenized texts in shape of [batch_size, seq_len].

        Returns:
            List[str]: The decoded texts. The first dimension indicates the batch.
        """


class NaiveTokenizer(Tokenizer):
    def __init__(self, config: Dict[str, Any]):
        self.text_folder_path = config.get("text_folder_path", None)
        self.token_folder_path = config.get("token_folder_path")
        self.vocab_file_path = os.path.join(self.token_folder_path, "vocab.json")
        self.itos = {}
        self.stoi = {}

        if not os.path.exists(self.vocab_file_path):
            # Build vocab from the text_folder_path
            print(f"Building vocabulary from {self.text_folder_path}")
            self._build_vocab_from_folder()
        else:
            # Load vocab from a file if it exists
            print(f"Loading vocabulary from {self.vocab_file_path}")
            with open(self.vocab_file_path, "r") as f:
                tokens_data = json.load(f)
                self.itos = {int(key): val for key, val in tokens_data["itos"].items()}
                self.stoi = {key: int(val) for key, val in tokens_data["stoi"].items()}

        # Add the <UNK> token for unknown characters
        if "<UNK>" not in self.stoi:
            unk_idx = len(self.itos)
            self.itos[unk_idx] = "<UNK>"
            self.stoi["<UNK>"] = unk_idx

    def _build_vocab_from_folder(self):
        if not self.text_folder_path or not os.path.exists(self.text_folder_path):
            raise ValueError("Need to specify the text_folder_path to build the tokenizer.")
        vocab = set()
        for filename in os.listdir(self.text_folder_path):
            with open(os.path.join(self.text_folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                vocab.update(list(text))

        self.itos = {i: c for i, c in enumerate(sorted(vocab))}
        self.stoi = {c: i for i, c in self.itos.items()}

        # Save this vocab to a file for future use
        with open(self.vocab_file_path, "w") as f:
            json.dump({"itos": self.itos, "stoi": self.stoi}, f)

    def vocab_size(self):
        return len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        tokens = [self.stoi.get(c, self.stoi["<UNK>"]) for c in text]
        return torch.LongTensor(tokens)

    def decode(self, tensor: torch.Tensor) -> str:
        decoded_text = "".join([self.itos.get(token_id.item(), "<UNK>") for token_id in tensor])
        return decoded_text


class BytePairTokenizer(Tokenizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vocab_size_config = self.config.get("vocab_size", 5000)
        self.token_folder_path = self.config.get("token_folder_path")
        self.vocab_file = os.path.join(self.token_folder_path, "vocab.json")
        self.merges_file = os.path.join(self.token_folder_path, "merges.txt")
        self.text_folder_path = self.config.get("text_folder_path")

        if os.path.exists(self.vocab_file) and os.path.exists(self.merges_file):
            # Load pre-trained tokenizer.
            self.tokenizer = ByteLevelBPETokenizer(self.vocab_file, self.merges_file)
        elif self.text_folder_path:
            # Initialize tokenizer for future training.
            self.tokenizer = ByteLevelBPETokenizer()
            self.min_frequency = self.config.get("min_frequency", 2)
        else:
            raise ValueError("Either provide metadata for a pretrained tokenizer or provide a path for the training data folder.")

    def train(self):
        if not self.text_folder_path:
            raise ValueError("Training data folder must be specified for training.")
        files = glob(os.path.join(self.text_folder_path, "*.txt"))
        print(f"Start to train the BPETokenizer with the data in {self.text_folder_path}")
        self.tokenizer.train(files=files, vocab_size=self.vocab_size_config, min_frequency=self.min_frequency)
        print(f"Finished training, write the data to {self.token_folder_path}")
        if not os.path.exists(self.token_folder_path):
            os.mkdir(self.token_folder_path)
        self.tokenizer.save_model(self.token_folder_path)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text).ids)

    def decode(self, tensor: Union[torch.Tensor, int, Sequence[int]]) -> str:
        # Convert tensor to Python list if it's a tensor
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().tolist()
        
        # If a single integer is passed, convert it to a list
        if isinstance(tensor, int):
            tensor = [tensor]
        
        # At this point, tensor should be a list or Sequence
        decoded_text = self.tokenizer.decode(tensor, skip_special_tokens=True)
        return decoded_text


def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser(description="Train a BytePairTokenizer")

    # Argument to specify the folder containing the training data
    parser.add_argument("--text_folder_path", "-d", type=str, default="./data", 
                        help="Folder containing the text files to train the tokenizer.")

    # Argument to specify where to save the tokenizer files
    parser.add_argument("--token_folder_path", "-t", type=str, default="./tokenizer/bpe", 
                        help="Folder to save the trained tokenizer files.")

    # Argument to specify the vocab size for the tokenizer
    parser.add_argument("--vocab_size", "-v", type=int, default=5000, 
                        help="Vocabulary size for the tokenizer.")

    # Argument to specify the minimum frequency for subwords
    parser.add_argument("--min_frequency", "-m", type=int, default=2, 
                        help="Minimum frequency for subwords in the training data.")

    return vars(parser.parse_args())


def main(args: Dict[str, str]):
    # Configurations for the BytePairTokenizer
    config = {
        "text_folder_path": args["text_folder_path"],
        "token_folder_path": args["token_folder_path"],
        "vocab_size": args["vocab_size"],
        "min_frequency": args["min_frequency"],
    }

    # Initialize and train the tokenizer
    tokenizer = BytePairTokenizer(config)
    tokenizer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)