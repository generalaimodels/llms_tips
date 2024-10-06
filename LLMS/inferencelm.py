import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextGenerator:
    """
    A class for generating text using a specified transformer model and tokenizer.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        cache_dir: Optional[str] = "./cache",
        device: Optional[str] = "cuda",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the TextGenerator with a model and tokenizer.

        Args:
            model_name_or_path (str): The model's name or path.
            tokenizer_name_or_path (Optional[str], optional): The tokenizer's name or path.
                If None, defaults to model_name_or_path.
            cache_dir (Optional[str], optional): Directory to cache the models. Defaults to "./cache".
            device (Optional[str], optional): Device to use, e.g., "cuda" or "cpu". Defaults to "cuda".
            **kwargs: Additional keyword arguments for model and tokenizer loading.

        Raises:
            ValueError: If the model or tokenizer cannot be loaded.
        """
        try:
            self.device = device
            device_map = "auto" if torch.cuda.is_available() and device == "cuda" else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **kwargs,
            )

            tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path

            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=True,
                **kwargs,
            )

            logger.info("Creating text generation pipeline...")
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                **kwargs,
            )
        except Exception as e:
            logger.exception("Failed to initialize TextGenerator.")
            raise ValueError(f"Failed to initialize TextGenerator: {e}") from e

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 500,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Generates text based on input messages.

        Args:
            messages (List[Dict[str, str]]): A list of messages, each a dict with "role" and "content".
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 500.
            do_sample (bool, optional): Whether to use sampling; uses greedy decoding if False. Defaults to False.
            **kwargs: Additional generation parameters.

        Returns:
            str: The generated text.

        Raises:
            ValueError: If text generation fails.
        """
        try:
            input_text = self._format_messages(messages)
            logger.info("Generating text...")
            output = self.generator(
                input_text,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                return_full_text=False,
                **kwargs,
            )
            return output[0]["generated_text"]
        except Exception as e:
            logger.exception("Failed to generate text.")
            raise ValueError(f"Failed to generate text: {e}") from e

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        """
        Formats messages for the model input.

        Args:
            messages (List[Dict[str, str]]): A list of messages.

        Returns:
            str: The formatted input text.
        """
        formatted_text = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_text += f"{role}: {content}\n"
        return formatted_text.strip()


# def main() -> None:
#     """
#     Main function to demonstrate the TextGenerator.
#     """
#     try:
#         # Initialize the text generator with desired parameters
#         generator = TextGenerator(
#             model_name_or_path="microsoft/Phi-3-mini-4k-instruct",
#             cache_dir="./cache",
#         )

#         # Define the messages or prompts
#         messages = [
#             {"role": "user", "content": "Create a funny joke about chickens."}
#         ]

#         # Generate the text
#         generated_text = generator.generate(messages)

#         # Output the generated text
#         print(generated_text)
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")


# if __name__ == "__main__":
#     main()