import logging
import argparse
import torch
from src.models import LiquidNeuralNetwork
from src.data_generator import TextDataGenerator
from src.trainer import TextTrainer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Liquid Neural Network Text Generation")
    parser.add_argument('--file_path', type=str, default='bible.txt', help='Path to the input text file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=128, help='Sequence length for training')
    parser.add_argument('--embed_size', type=int, default=128, help='Embedding size')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 128, 128, 64, 64, 32], help='Hidden layer sizes')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--print_every', type=int, default=100, help='Print status every n steps')
    parser.add_argument('--gen_length', type=int, default=100, help='Length of generated text')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to load checkpoint from')
    return parser.parse_args()

def main():
    args = parse_arguments()

    try:
        data_generator = TextDataGenerator(args.file_path, args.batch_size, args.sequence_length)
        logging.info(f"Data generator initialized. Vocab size: {data_generator.vocab_size}")

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model = LiquidNeuralNetwork(
                vocab_size=checkpoint['vocab_size'],
                embed_size=checkpoint['embedding_size'],
                hidden_sizes=checkpoint['hidden_sizes'],
                sequence_length=checkpoint['sequence_length']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded checkpoint from {args.checkpoint}")
        else:
            model = LiquidNeuralNetwork(
                vocab_size=data_generator.vocab_size,
                embed_size=args.embed_size,
                hidden_sizes=args.hidden_sizes,
                sequence_length=args.sequence_length
            )
            logging.info(f"Created new model with vocab_size={data_generator.vocab_size}, embed_size={args.embed_size}, hidden_sizes={args.hidden_sizes}, sequence_length={args.sequence_length}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        trainer = TextTrainer(model, data_generator, args.learning_rate, device)

        if args.checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logging.info("Starting continual learning process...")
        trainer.continual_train_and_generate(args.num_steps, args.print_every, args.gen_length)

        logging.info("Performing final evaluation...")
        trainer.evaluate()

    except FileNotFoundError:
        logging.error(f"Input file not found: {args.file_path}")
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA out of memory. Try reducing batch size or model size.")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()