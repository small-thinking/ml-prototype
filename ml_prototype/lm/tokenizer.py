import abc
import json
import glob
import os
from typing import Any, Dict, List

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
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode the input texts.

        Args:
            texts (List[str]): The input texts. The first dimension indicates the batch.

        Returns:
            torch.Tensor: The tokenized texts in shape of [batch_size, seq_len].
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
    def __init__(self, config: Dict[str, Any], token_file: str):
        super().__init__(config)
        with open(token_file, "r") as f:
            tokens_data = json.load(f)
            self.itos = {int(key): val for key, val in tokens_data["itos"].items()}
            self.stoi = {key: int(val) for key, val in tokens_data["stoi"].items()}

    def vocab_size(self):
        return len(self.vocab)

    def encode(self, texts: str) -> torch.Tensor:
        """Encode the input texts."""
        tokens = []
        for text in texts:
            tokens.append([self.stoi[c] for c in text])
        return torch.LongTensor(tokens)

    def decode(self, tensor: torch.Tensor) -> str:
        """Decode the tensor back to text."""
        decoded_text = "".join([self.itos[token_id.item()] for token_id in tensor])
        return decoded_text


class BytePairTokenizer(Tokenizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vocab_file = self.config.get("vocab_file")
        self.merges_file = self.config.get("merges_file")
        self.train_data_folder = self.config.get("train_data_folder")

        if self.vocab_file and self.merges_file:
            # Load pre-trained tokenizer.
            self.tokenizer = ByteLevelBPETokenizer(self.vocab_file, self.merges_file)
        elif self.train_data_folder:
            # Train tokenizer from scratch.
            self.tokenizer = ByteLevelBPETokenizer()
            self.vocab_size = self.config.get("vocab_size", 5000)
            self.min_frequency = self.config.get("min_frequency", 2)
        else:
            raise ValueError(
                "Either provide the metadata of a pretrained tokenizer or provide path of the training data folder."
            )

    def train(self):
        files = glob.glob(os.path.join(self.train_data_folder, "*.txt"))
        self.tokenizer.train(files=files, vocab_size=self.vocab_size, min_frequency=self.min_frequency)
        self.tokenizer.save_model(".", self.config.get("tokenizer_name"))

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, texts: List[str]) -> torch.Tensor:
        return torch.tensor([self.tokenizer.encode(text).ids for text in texts])

    def decode(self, tensor: torch.Tensor) -> List[str]:
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in tensor.tolist()]