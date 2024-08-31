import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Tuple

class TextTrainer:
    def __init__(self, model: nn.Module, data_generator, learning_rate: float = 0.001, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.data_generator = data_generator
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device
        logging.info(f"Using device: {self.device}")

    def train_step(self) -> Tuple[float, float]:
        x, y_true = self.data_generator.generate_batch()
        x, y_true = x.to(self.device), y_true.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        y_pred, _, _ = self.model(x)

        loss = self.criterion(y_pred.view(-1, self.model.vocab_size), y_true.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        perplexity = torch.exp(loss)
        return loss.item(), perplexity.item()

    def generate_text(self, start_text: str, gen_length: int, temperature: float = 0.5) -> str:
        return self.model.generate(self.data_generator, start_text, gen_length, temperature)

    def train(self, num_steps: int, print_every: int = 100):
        try:
            for step in range(1, num_steps + 1):
                loss, perplexity = self.train_step()

                if step % print_every == 0:
                    logging.info(f"Step {step}/{num_steps}")
                    logging.info(f"Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")

                if step % (print_every * 10) == 0:
                    self.adjust_learning_rate()

        except KeyboardInterrupt:
            logging.info("Training interrupted by user.")
        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
        finally:
            logging.info("Saving model checkpoint...")
            self.save_checkpoint()

    def interactive_mode(self):
        print("Entering interactive mode. Type 'exit' to quit.")
        conversation_history = []
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            conversation_history.append(f"{user_input}")
            context = "\n".join(conversation_history[-5:])

            model_response = self.generate_text(context, gen_length=50)
            
            clean_response = model_response.replace(f"{context}", "").strip()
            
            print(f"Model: {clean_response}")

            conversation_history.append(f"{clean_response}")

            self.update_model_with_new_text(user_input + "\n" + clean_response)

        print("Exiting interactive mode.")

    def update_model_with_new_text(self, new_text: str) -> None:
        old_vocab_size = self.model.vocab_size
        self.data_generator.add_new_text(new_text)
        
        if self.data_generator.vocab_size > old_vocab_size:
            self.model.expand_embedding(self.data_generator.vocab_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'])
            logging.info("Model updated with new vocabulary")

        self.train_step()

    def adjust_learning_rate(self, decay_factor: float = 0.1) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay_factor
        logging.info(f"Learning rate adjusted to {self.optimizer.param_groups[0]['lr']:.6f}")

    def save_checkpoint(self, filename: str = 'model_checkpoint.pth') -> None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': self.model.vocab_size,
            'embedding_size': self.model.embed_size,
            'hidden_sizes': self.model.hidden_sizes,
            'sequence_length': self.model.sequence_length
        }
        torch.save(checkpoint, filename)
        logging.info(f"Model checkpoint saved to {filename}")

    def load_checkpoint(self, filename: str = 'model_checkpoint.pth') -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.vocab_size = checkpoint['vocab_size']
        self.model.embed_size = checkpoint['embedding_size']
        self.model.hidden_sizes = checkpoint['hidden_sizes']
        self.model.sequence_length = checkpoint['sequence_length']
        logging.info(f"Model checkpoint loaded from {filename}")

    def evaluate(self, num_batches: int = 10) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_perplexity = 0

        with torch.no_grad():
            for _ in range(num_batches):
                x, y_true = self.data_generator.generate_batch()
                x, y_true = x.to(self.device), y_true.to(self.device)

                y_pred, _, _ = self.model(x)
                loss = self.criterion(y_pred.view(-1, self.model.vocab_size), y_true.view(-1))
                perplexity = torch.exp(loss)

                total_loss += loss.item()
                total_perplexity += perplexity.item()

        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        logging.info(f"Evaluation - Avg Loss: {avg_loss:.4f}, Avg Perplexity: {avg_perplexity:.4f}")
        return avg_loss, avg_perplexity