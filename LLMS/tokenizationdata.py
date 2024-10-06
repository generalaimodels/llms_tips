from typing import List, Dict, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
import logging

# Set up logging format for error handling and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TokenizerPipeline:
    """
    A modular and reusable tokenization pipeline that works with Hugging Face's pretrained tokenizers.
    It handles tokenization, padding, truncation, batch processing, and encoding/decoding of text data.
    """

    def __init__(self, model_name: str):
        """
        Initialize the tokenizer for a given model.
        
        :param model_name: The pretrained model name or path compatible with Hugging Face.
        """
        try:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
            logging.info(f"Tokenizer successfully loaded for model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading tokenizer for model {model_name}: {str(e)}")
            raise

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 128,
        padding: Union[bool, str] = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt',
        add_special_tokens: bool = True
    ) -> BatchEncoding:
        """
        Tokenize a list of sentences with specified parameters. Handles batch-wise tokenization.
        
        :param texts: List of sentences to tokenize.
        :param max_length: Maximum length for token sequences (padded/truncated to this length).
        :param padding: Padding strategy ('max_length' or 'longest').
        :param truncation: Whether to truncate sequences longer than max_length.
        :param return_tensors: Format for returned tensor ('pt' for PyTorch, 'tf' for TensorFlow, 'np' for NumPy, or None).
        :param add_special_tokens: Whether to encode with special tokens based on the model.
        :return: BatchEncoding object containing tokens and related data.
        """
        try:
            tokenized_output: BatchEncoding = self.tokenizer(
                texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens
            )
            logging.info(f"Tokenized {len(texts)} sentences successfully.")
            return tokenized_output
        except Exception as e:
            logging.error(f"Error during tokenization: {str(e)}")
            raise

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back into human-readable text.

        :param token_ids: Token IDs to decode.
        :param skip_special_tokens: Whether to remove special tokens in the decoded output.
        :return: Decoded string.
        """
        try:
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            logging.info(f"Decoded token IDs into text successfully.")
            return decoded_text
        except Exception as e:
            logging.error(f"Error during decoding: {str(e)}")
            raise

    def batch_decode(self, batch_token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID sequences into human-readable text.

        :param batch_token_ids: List of token ID sequences.
        :param skip_special_tokens: Whether to remove special tokens in the decoded output.
        :return: List of decoded strings.
        """
        try:
            decoded_texts = self.tokenizer.batch_decode(batch_token_ids, skip_special_tokens=skip_special_tokens)
            logging.info(f"Batch decoded {len(batch_token_ids)} token sequences successfully.")
            return decoded_texts
        except Exception as e:
            logging.error(f"Error during batch decoding: {str(e)}")
            raise


# # Example usage of the pipeline
# if __name__ == "__main__":
#     # Initialize with a popular pre-trained tokenizer
#     model_name = "bert-base-uncased"  # Can be changed to any other model, e.g., GPT-2, Roberta
#     tokenizer_pipeline = TokenizerPipeline(model_name)

#     # Input data
#     sentences = [
#         "Hello, how are you?",
#         "I love using Hugging Face's tokenizers!",
#         "This is a test sentence for tokenizer pipelines."
#     ]

#     # Tokenize the input sentences
#     tokenized_output = tokenizer_pipeline.tokenize(
#         texts=sentences,
#         max_length=10,  # Example of truncation/padding
#         padding='max_length',
#         truncation=True
#     )

#     # Print tokenized output (token IDs)
#     print("Tokenized Output (IDs):", tokenized_output['input_ids'])

#     # Decode the first tokenized sentence back to text
#     decoded_sentence = tokenizer_pipeline.decode(tokenized_output['input_ids'][0])
#     print("\nDecoded Sentence:", decoded_sentence)

#     # Batch decode all tokenized sentences
#     decoded_batch = tokenizer_pipeline.batch_decode(tokenized_output['input_ids'])
#     print("\nBatch Decoded Sentences:", decoded_batch)