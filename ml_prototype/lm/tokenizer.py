import abc
import json
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
