from typing import List, Dict, Any
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    pipeline,
    Pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    Text2TextGenerationPipeline,
    TFAutoModelForSeq2SeqLM,
)
from transformers.pipelines.pt_utils import KeyDataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


class NLPTaskHandler:
    """
    A class to handle various NLP tasks such as text classification, zero-shot learning,
    sentiment analysis, and text generation in a generalized and robust manner.
    """

    def __init__(
        self,
        dataset_name: str,
        text_column: str = "text",
        label_column: str = "label",
        device: str = "cpu",
    ) -> None:
        """
        Initialize the NLPTaskHandler with dataset and device information.

        Args:
            dataset_name (str): The name of the dataset to load.
            text_column (str, optional): The name of the text column in the dataset. Defaults to "text".
            label_column (str, optional): The name of the label column in the dataset. Defaults to "label".
            device (str, optional): The device to run computations on. Defaults to "cpu".
        """
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.label_column = label_column
        self.device = device
        self.data = self.load_data()
        self.model = None
        self.pipe = None

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> DatasetDict:
        """
        Load the dataset using the Hugging Face Datasets library.

        Returns:
            DatasetDict: The loaded dataset.
        """
        try:
            data = load_dataset(self.dataset_name)
            self.logger.info(f"Dataset '{self.dataset_name}' loaded successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load dataset '{self.dataset_name}': {e}")
            raise

    def load_text_classification_model(self, model_path: str) -> None:
        """
        Load a text classification model into the pipeline.

        Args:
            model_path (str): The Hugging Face model path to load.
        """
        try:
            self.pipe = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                return_all_scores=True,
                device=self.device,
            )
            self.logger.info(f"Model '{model_path}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load model '{model_path}': {e}")
            raise

    def run_text_classification_inference(self, split: str = "test") -> List[int]:
        """
        Run inference using the loaded text classification model.

        Args:
            split (str, optional): The dataset split to run inference on. Defaults to "test".

        Returns:
            List[int]: The predicted labels.
        """
        if self.pipe is None:
            self.logger.error("Pipeline is not initialized. Load a model first.")
            raise ValueError("Pipeline is not initialized.")

        y_pred = []
        dataset = self.data[split]
        self.logger.info(f"Running inference on the '{split}' split.")

        try:
            for output in tqdm(
                self.pipe(KeyDataset(dataset, self.text_column)),
                total=len(dataset),
                desc="Inference",
            ):
                # Assuming binary classification with label mapping
                scores = [score_dict["score"] for score_dict in output]
                assignment = np.argmax(scores)
                y_pred.append(assignment)
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise

        return y_pred

    def evaluate_performance(
        self, y_true: List[int], y_pred: List[int], target_names: List[str]
    ) -> None:
        """
        Evaluate the performance of the model and print the classification report.

        Args:
            y_true (List[int]): The true labels.
            y_pred (List[int]): The predicted labels.
            target_names (List[str]): The names of the target classes.
        """
        try:
            performance = classification_report(
                y_true, y_pred, target_names=target_names
            )
            print(performance)
            self.logger.info("Performance evaluation completed.")
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

    def encode_text(self, model_name: str, split: str = "train") -> np.ndarray:
        """
        Encode text data into embeddings using a sentence transformer model.

        Args:
            model_name (str): The name of the sentence transformer model.
            split (str, optional): The dataset split to encode. Defaults to "train".

        Returns:
            np.ndarray: The text embeddings.
        """
        try:
            model = SentenceTransformer(model_name, device=self.device)
            self.logger.info(f"SentenceTransformer model '{model_name}' loaded.")
            embeddings = model.encode(
                self.data[split][self.text_column],
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            self.logger.info(f"Text data from '{split}' split encoded successfully.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            raise

    def train_logistic_regression(
        self, embeddings: np.ndarray, labels: List[int]
    ) -> LogisticRegression:
        """
        Train a Logistic Regression classifier on the provided embeddings.

        Args:
            embeddings (np.ndarray): The text embeddings.
            labels (List[int]): The labels corresponding to the embeddings.

        Returns:
            LogisticRegression: The trained Logistic Regression model.
        """
        try:
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(embeddings, labels)
            self.logger.info("Logistic Regression model trained successfully.")
            return clf
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def cosine_similarity_classification(
        self, test_embeddings: np.ndarray, train_embeddings: np.ndarray, labels: List[int]
    ) -> List[int]:
        """
        Perform classification by finding the nearest centroid using cosine similarity.

        Args:
            test_embeddings (np.ndarray): The embeddings of the test data.
            train_embeddings (np.ndarray): The embeddings of the train data.
            labels (List[int]): The labels corresponding to the train embeddings.

        Returns:
            List[int]: The predicted labels for the test data.
        """
        try:
            df = pd.DataFrame(train_embeddings)
            df[self.label_column] = labels
            centroids = df.groupby(self.label_column).mean().values
            self.logger.info("Calculated centroids for each class.")

            closest, _ = pairwise_distances_argmin_min(
                test_embeddings, centroids, metric="cosine"
            )
            self.logger.info("Assigned labels based on nearest centroid.")
            return closest
        except Exception as e:
            self.logger.error(f"Cosine similarity classification failed: {e}")
            raise

    def zero_shot_classification(
        self, model_name: str, split: str = "test", candidate_labels: List[str] = None
    ) -> List[int]:
        """
        Perform zero-shot classification using cosine similarity with label embeddings.

        Args:
            model_name (str): The name of the sentence transformer model.
            split (str, optional): The dataset split to classify. Defaults to "test".
            candidate_labels (List[str], optional): The list of label descriptions. Defaults to None.

        Returns:
            List[int]: The predicted labels.
        """
        if candidate_labels is None:
            candidate_labels = ["A negative review", "A positive review"]

        try:
            model = SentenceTransformer(model_name, device=self.device)
            self.logger.info(f"SentenceTransformer model '{model_name}' loaded.")

            # Encode the test data and labels
            test_embeddings = model.encode(
                self.data[split][self.text_column],
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            label_embeddings = model.encode(
                candidate_labels, show_progress_bar=False, convert_to_numpy=True
            )
            self.logger.info("Encoded test data and candidate labels.")

            # Compute similarities and predict labels
            similarities = cosine_similarity(test_embeddings, label_embeddings)
            y_pred = np.argmax(similarities, axis=1)
            self.logger.info("Zero-shot classification completed.")
            return y_pred
        except Exception as e:
            self.logger.error(f"Zero-shot classification failed: {e}")
            raise

    def text_generation_inference(
        self, model_name: str, prompt_template: str, split: str = "test"
    ) -> List[int]:
        """
        Run text generation inference using a sequence-to-sequence model.

        Args:
            model_name (str): The name of the seq2seq model.
            prompt_template (str): The prompt template to prepend to each text.
            split (str, optional): The dataset split to run inference on. Defaults to "test".

        Returns:
            List[int]: The predicted labels based on generated text.
        """
        try:
            # Load the model
            self.pipe = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
            )
            self.logger.info(f"Seq2Seq model '{model_name}' loaded.")

            # Prepare data with prompts
            self.data = self.data.map(
                lambda example: {
                    "prompted_text": prompt_template + example[self.text_column]
                }
            )
            self.logger.info("Data has been prepared with prompts.")

            # Run inference
            y_pred = []
            dataset = self.data[split]
            for output in tqdm(
                self.pipe(KeyDataset(dataset, "prompted_text")),
                total=len(dataset),
                desc="Text Generation Inference",
            ):
                generated_text = output[0]["generated_text"].strip().lower()
                if "negative" in generated_text:
                    y_pred.append(0)
                elif "positive" in generated_text:
                    y_pred.append(1)
                else:
                    # Handle unexpected outputs
                    y_pred.append(-1)
            self.logger.info("Text generation inference completed.")
            return y_pred
        except Exception as e:
            self.logger.error(f"Text generation inference failed: {e}")
            raise


# if __name__ == "__main__":
#     # Example usage
#     nlp_handler = NLPTaskHandler(
#         dataset_name="My_dataset",
#         text_column="text",
#         label_column="label",
#         device=0,  # Use GPU if available
#     )

#     # Load and run text classification model
#     nlp_handler.load_text_classification_model(
#         model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"
#     )
#     y_pred = nlp_handler.run_text_classification_inference()
#     y_true = nlp_handler.data["test"][nlp_handler.label_column]
#     nlp_handler.evaluate_performance(
#         y_true=y_true,
#         y_pred=y_pred,
#         target_names=["Negative Review", "Positive Review"],
#     )

#     # Encode text data and train logistic regression
#     train_embeddings = nlp_handler.encode_text(
#         model_name="sentence-transformers/all-mpnet-base-v2", split="train"
#     )
#     test_embeddings = nlp_handler.encode_text(
#         model_name="sentence-transformers/all-mpnet-base-v2", split="test"
#     )
#     clf = nlp_handler.train_logistic_regression(
#         embeddings=train_embeddings,
#         labels=nlp_handler.data["train"][nlp_handler.label_column],
#     )
#     y_pred = clf.predict(test_embeddings)
#     nlp_handler.evaluate_performance(
#         y_true=nlp_handler.data["test"][nlp_handler.label_column],
#         y_pred=y_pred,
#         target_names=["Negative Review", "Positive Review"],
#     )

#     # Cosine similarity classification
#     y_pred = nlp_handler.cosine_similarity_classification(
#         test_embeddings=test_embeddings,
#         train_embeddings=train_embeddings,
#         labels=nlp_handler.data["train"][nlp_handler.label_column],
#     )
#     nlp_handler.evaluate_performance(
#         y_true=nlp_handler.data["test"][nlp_handler.label_column],
#         y_pred=y_pred,
#         target_names=["Negative Review", "Positive Review"],
#     )

#     # Zero-shot classification
#     y_pred = nlp_handler.zero_shot_classification(
#         model_name="sentence-transformers/all-mpnet-base-v2"
#     )
#     nlp_handler.evaluate_performance(
#         y_true=nlp_handler.data["test"][nlp_handler.label_column],
#         y_pred=y_pred,
#         target_names=["Negative Review", "Positive Review"],
#     )

#     # Text generation inference
#     y_pred = nlp_handler.text_generation_inference(
#         model_name="google/flan-t5-small",
#         prompt_template="Is the following sentence positive or negative? ",
#     )
#     y_true = nlp_handler.data["test"][nlp_handler.label_column]
#     nlp_handler.evaluate_performance(
#         y_true=y_true,
#         y_pred=y_pred,
#         target_names=["Negative Review", "Positive Review"],
#     )