import abc
import os
from typing import Any, Dict, List

import torch


class Tokenizer(abc.ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode the input texts.

        Args:
            texts (List[str]): The input texts. The first dimension indicates the batch.

        Returns:
            torch.Tensor: The tokenized texts in shape of [batch_size, seq_len].
        """
        raise NotImplementedError

    def decode(self, tensor: torch.Tensor) -> List[str]:
        """Decode the tensor back to text.

        Args:
            tensor (torch.Tensor): The tokenized texts in shape of [batch_size, seq_len].

        Returns:
            List[str]: The decoded texts. The first dimension indicates the batch.
        """


class NaiveTokenizer(Tokenizer):
    def __init__(self, config: Dict[str, Any], doc_file_path: str):
        super().__init__(config)
        lines = open(os.path.expanduser(doc_file_path)).read()
        vocab = sorted(list(set(lines)))
        self.itos = {i: word for i, word in enumerate(vocab)}
        self.stoi = {word: i for i, word in enumerate(vocab)}

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode the input texts."""
        tokens = []
        for text in texts:
            tokens.append([self.stoi[c] for c in text])
        return torch.LongTensor(tokens)  # Change to LongTensor

    def decode(self, tensor: torch.Tensor) -> List[str]:
        """Decode the tensor back to text."""
        decoded_texts = []
        for sequence in tensor:
            decoded_texts.append("".join([self.itos[i.item()] for i in sequence]))
        return decoded_texts
