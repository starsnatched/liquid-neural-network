import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from typing import Tuple
import time

class TextTrainer:
    def __init__(self, model: nn.Module, data_generator, learning_rate: float = 0.001, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.data_generator = data_generator
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-6)
        self.device = device
        logging.info(f"Using device: {self.device}")

    def train_step(self) -> Tuple[float, float]:
        x, y_true = self.data_generator.generate_batch()
        x, y_true = x.to(self.device), y_true.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            y_pred = self.model(x)
        except RuntimeError as e:
            logging.error(f"Error in forward pass: {str(e)}")
            return float('inf'), float('inf')

        loss = self.criterion(y_pred.view(-1, self.model.vocab_size), y_true.view(-1))
        
        try:
            loss.backward()
        except RuntimeError as e:
            logging.error(f"Error in backward pass: {str(e)}")
            return float('inf'), float('inf')
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()

        perplexity = torch.exp(loss)
        return loss.item(), perplexity.item()

    def train(self, num_steps: int, print_every: int = 100, eval_every: int = 1000):
        best_loss = float('inf')
        steps_without_improvement = 0
        start_time = time.time()
        
        try:
            for step in range(1, num_steps + 1):
                loss, perplexity = self.train_step()

                if step % print_every == 0:
                    elapsed_time = time.time() - start_time
                    logging.info(f"Step {step}/{num_steps} ({elapsed_time:.2f}s)")
                    logging.info(f"Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")
                    logging.info(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
                    
                if step % eval_every == 0:
                    eval_loss, eval_perplexity = self.evaluate()
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        steps_without_improvement = 0
                        self.save_checkpoint(filename='best_model_checkpoint.pth')
                    else:
                        steps_without_improvement += eval_every
                        if steps_without_improvement >= 5000:
                            logging.info("Early stopping triggered")
                            break

        except KeyboardInterrupt:
            logging.info("Training interrupted by user.")
        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
        finally:
            logging.info("Saving final model checkpoint...")
            self.save_checkpoint(filename='final_model_checkpoint.pth')

    def evaluate(self, num_batches: int = 10) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_perplexity = 0

        with torch.no_grad():
            for _ in range(num_batches):
                x, y_true = self.data_generator.generate_batch()
                x, y_true = x.to(self.device), y_true.to(self.device)

                y_pred = self.model(x)
                loss = self.criterion(y_pred.view(-1, self.model.vocab_size), y_true.view(-1))
                perplexity = torch.exp(loss)

                total_loss += loss.item()
                total_perplexity += perplexity.item()

        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        logging.info(f"Evaluation - Avg Loss: {avg_loss:.4f}, Avg Perplexity: {avg_perplexity:.4f}")
        return avg_loss, avg_perplexity

    def save_checkpoint(self, filename: str = 'model_checkpoint.pth') -> None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model.get_config()
        }
        torch.save(checkpoint, filename)
        logging.info(f"Model checkpoint saved to {filename}")

    def load_checkpoint(self, filename: str = 'model_checkpoint.pth') -> None:
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info(f"Model checkpoint loaded from {filename}")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")