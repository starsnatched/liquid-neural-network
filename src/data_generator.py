import torch
import numpy as np
import logging
from collections import deque
from typing import List, Tuple

class TextDataGenerator:
    def __init__(self, file_path: str, batch_size: int, sequence_length: int, buffer_size: int = 10000):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.text_buffer = deque(maxlen=buffer_size)
        self.chars = set()
        self.load_initial_text()
        self.update_vocab()
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def load_initial_text(self) -> None:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                initial_text = file.read(self.buffer_size)
                self.text_buffer.extend(initial_text)
                self.chars.update(set(initial_text))
        except IOError as e:
            logging.error(f"Error reading file: {str(e)}")
            raise

    def update_vocab(self) -> None:
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def add_new_text(self, new_text: str) -> None:
        new_chars = set(new_text) - self.chars
        if new_chars:
            self.chars.update(new_chars)
            self.update_vocab()
            logging.info(f"Added {len(new_chars)} new characters to vocabulary")
        self.text_buffer.extend(new_text)

    def char_to_tensor(self, char: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx.get(char, 0)], dtype=torch.long)

    def generate_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(self.batch_size, self.sequence_length, dtype=torch.long)
        y = torch.zeros(self.batch_size, self.sequence_length, dtype=torch.long)

        text_list = list(self.text_buffer)
        valid_start_indices = len(text_list) - self.sequence_length - 1

        if valid_start_indices < self.batch_size:
            logging.warning("Not enough data to generate a full batch. Consider increasing buffer_size or reducing batch_size.")
            return x[:valid_start_indices], y[:valid_start_indices]

        start_indices = np.random.randint(0, valid_start_indices, size=self.batch_size)

        for i, start_idx in enumerate(start_indices):
            chunk = text_list[start_idx:start_idx + self.sequence_length + 1]
            x[i] = torch.tensor([self.char_to_idx.get(ch, 0) for ch in chunk[:-1]])
            y[i] = torch.tensor([self.char_to_idx.get(ch, 0) for ch in chunk[1:]])

        return x, y

    def decode(self, tensor: torch.Tensor) -> str:
        return ''.join([self.idx_to_char.get(idx.item(), '') for idx in tensor])

    def __len__(self) -> int:
        return len(self.text_buffer)