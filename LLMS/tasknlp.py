import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline, Pipeline
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_path: str
    task: str
    device: Optional[int] = None


@dataclass
class DatasetConfig:
    name: str
    split: List[str] = field(default_factory=lambda: ["train", "validation", "test"])
    data_dir: Optional[str] = None
    cache_dir: Optional[str] = None


class TextClassifier:
    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        prompt: Optional[str] = None
    ):
        """
        Initialize the TextClassifier with model and dataset configurations.

        Args:
            model_config (ModelConfig): Configuration for the model.
            dataset_config (DatasetConfig): Configuration for the dataset.
            prompt (Optional[str], optional): Prompt for text generation tasks. Defaults to None.
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.prompt = prompt
        self.dataset: Optional[DatasetDict] = None
        self.pipe: Optional[Pipeline] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.label_names: Optional[List[str]] = None

        logger.info("TextClassifier initialized with model '%s' and dataset '%s'.",
                    self.model_config.model_path, self.dataset_config.name)

    def load_data(self) -> None:
        """Load the dataset based on the provided configuration."""
        try:
            self.dataset = load_dataset(
                self.dataset_config.name,
                data_dir=self.dataset_config.data_dir,
                cache_dir=self.dataset_config.cache_dir
            )
            logger.info("Dataset loaded successfully with splits: %s",
                        list(self.dataset.keys()))
        except Exception as e:
            logger.error("Failed to load dataset: %s", e)
            raise

    def initialize_pipeline(self) -> None:
        """Initialize the Hugging Face pipeline based on the task."""
        try:
            self.pipe = pipeline(
                task=self.model_config.task,
                model=self.model_config.model_path,
                tokenizer=self.model_config.model_path,
                device=self.model_config.device if self.model_config.device is not None else -1
            )
            logger.info("Pipeline initialized for task '%s'.", self.model_config.task)
        except Exception as e:
            logger.error("Failed to initialize pipeline: %s", e)
            raise

    def run_inference(
        self,
        split: str = "test",
        text_column: str = "text",
        generation_column: Optional[str] = None
    ) -> List[int]:
        """
        Run inference on the specified dataset split.

        Args:
            split (str, optional): Dataset split to use. Defaults to "test".
            text_column (str, optional): Column name for input text. Defaults to "text".
            generation_column (Optional[str], optional): Column name for text generation tasks. Defaults to None.

        Returns:
            List[int]: List of predicted labels.
        """
        if self.dataset is None:
            logger.error("Dataset not loaded. Call load_data() before run_inference().")
            raise ValueError("Dataset not loaded.")

        if self.pipe is None:
            logger.error("Pipeline not initialized. Call initialize_pipeline() before run_inference().")
            raise ValueError("Pipeline not initialized.")

        try:
            texts = self.dataset[split][text_column]
            if generation_column and self.prompt:
                texts = [self.prompt + text for text in texts]
                input_texts = generation_column
            else:
                input_texts = text_column

            y_pred = []
            key_dataset = self.dataset[split][text_column]
            for output in tqdm(self.pipe(key_dataset), total=len(self.dataset[split])):
                if self.model_config.task.startswith("text2text"):
                    generated_text = output[0].get("generated_text", "").lower()
                    y_pred.append(0 if "negative" in generated_text else 1)
                else:
                    # Assuming the pipeline returns a list of scores
                    scores = output if isinstance(output, list) else output.get("scores", [])
                    pred = np.argmax([score['score'] for score in scores])
                    y_pred.append(pred)

            logger.info("Inference completed on split '%s'.", split)
            return y_pred
        except Exception as e:
            logger.error("Failed during inference: %s", e)
            raise

    def evaluate_performance(
        self,
        split: str = "test",
        y_pred: Optional[List[int]] = None,
        label_column: str = "label",
        target_names: Optional[List[str]] = None
    ) -> None:
        """
        Evaluate the performance of the predictions.

        Args:
            split (str, optional): Dataset split to evaluate. Defaults to "test".
            y_pred (Optional[List[int]], optional): Predicted labels. Defaults to None.
            label_column (str, optional): Column name for true labels. Defaults to "label".
            target_names (Optional[List[str]], optional): Names of the target classes. Defaults to None.
        """
        if self.dataset is None:
            logger.error("Dataset not loaded. Call load_data() before evaluate_performance().")
            raise ValueError("Dataset not loaded.")

        if y_pred is None:
            logger.error("Predictions not provided.")
            raise ValueError("Predictions not provided.")

        try:
            y_true = self.dataset[split][label_column]
            if target_names is None and self.label_names is not None:
                target_names = self.label_names
            elif target_names is None:
                target_names = [str(i) for i in sorted(set(y_true))]

            report = classification_report(
                y_true,
                y_pred,
                target_names=target_names
            )
            logger.info("Classification Report:\n%s", report)
            print(report)
        except Exception as e:
            logger.error("Failed during evaluation: %s", e)
            raise

    def encode_embeddings(
        self,
        split: str = "train",
        text_column: str = "text"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode text data into embeddings using SentenceTransformer.

        Args:
            split (str, optional): Dataset split to use. Defaults to "train".
            text_column (str, optional): Column name for input text. Defaults to "text".

        Returns:
            Tuple[np.ndarray, np.ndarray]: Train embeddings and corresponding labels.
        """
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("SentenceTransformer model loaded for embeddings.")

            texts = self.dataset[split][text_column]
            labels = self.dataset[split]["label"]
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info("Embeddings encoded for split '%s'.", split)
            return embeddings, np.array(labels)
        except Exception as e:
            logger.error("Failed to encode embeddings: %s", e)
            raise

    def train_logistic_regression(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray
    ) -> None:
        """
        Train a Logistic Regression classifier on the provided embeddings.

        Args:
            train_embeddings (np.ndarray): Training embeddings.
            train_labels (np.ndarray): Training labels.
        """
        try:
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
            self.classifier.fit(train_embeddings, train_labels)
            logger.info("Logistic Regression model trained successfully.")
        except Exception as e:
            logger.error("Failed to train Logistic Regression: %s", e)
            raise

    def predict_with_classifier(
        self,
        test_embeddings: np.ndarray
    ) -> List[int]:
        """
        Predict labels using the trained Logistic Regression classifier.

        Args:
            test_embeddings (np.ndarray): Test embeddings.

        Returns:
            List[int]: Predicted labels.
        """
        if self.classifier is None:
            logger.error("Classifier not trained. Call train_logistic_regression() first.")
            raise ValueError("Classifier not trained.")

        try:
            predictions = self.classifier.predict(test_embeddings).tolist()
            logger.info("Predictions made using Logistic Regression classifier.")
            return predictions
        except Exception as e:
            logger.error("Failed to make predictions with classifier: %s", e)
            raise

    def average_embeddings(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        num_features: int = 768
    ) -> np.ndarray:
        """
        Average the embeddings of all documents in each target label.

        Args:
            train_embeddings (np.ndarray): Training embeddings.
            train_labels (np.ndarray): Training labels.
            num_features (int, optional): Number of features in embeddings. Defaults to 768.

        Returns:
            np.ndarray: Averaged embeddings for each label.
        """
        try:
            df = pd.DataFrame(np.hstack([train_embeddings, train_labels.reshape(-1, 1)]))
            self.label_names = sorted(df.iloc[:, -1].unique())
            averaged_embeddings = df.groupby(df.columns[-1]).mean().iloc[:, :-1].to_numpy()
            logger.info("Averaged embeddings computed for each label.")
            return averaged_embeddings
        except Exception as e:
            logger.error("Failed to average embeddings: %s", e)
            raise

    def predict_with_cosine_similarity(
        self,
        test_embeddings: np.ndarray,
        averaged_embeddings: np.ndarray
    ) -> List[int]:
        """
        Predict labels based on cosine similarity between test embeddings and averaged label embeddings.

        Args:
            test_embeddings (np.ndarray): Test embeddings.
            averaged_embeddings (np.ndarray): Averaged embeddings for each label.

        Returns:
            List[int]: Predicted labels.
        """
        try:
            sim_matrix = cosine_similarity(test_embeddings, averaged_embeddings)
            predictions = np.argmax(sim_matrix, axis=1).tolist()
            logger.info("Predictions made using cosine similarity.")
            return predictions
        except Exception as e:
            logger.error("Failed to predict with cosine similarity: %s", e)
            raise

    def create_label_embeddings(
        self,
        labels: List[str]
    ) -> np.ndarray:
        """
        Create embeddings for given label descriptions.

        Args:
            labels (List[str]): List of label descriptions.

        Returns:
            np.ndarray: Embeddings for each label.
        """
        try:
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                logger.info("SentenceTransformer model loaded for label embeddings.")

            label_embeddings = self.embedding_model.encode(
                labels,
                convert_to_numpy=True
            )
            logger.info("Label embeddings created.")
            return label_embeddings
        except Exception as e:
            logger.error("Failed to create label embeddings: %s", e)
            raise

    def predict_with_label_embeddings(
        self,
        test_embeddings: np.ndarray,
        label_embeddings: np.ndarray
    ) -> List[int]:
        """
        Predict labels based on similarity to label embeddings.

        Args:
            test_embeddings (np.ndarray): Test embeddings.
            label_embeddings (np.ndarray): Embeddings of labels.

        Returns:
            List[int]: Predicted labels.
        """
        try:
            sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
            predictions = np.argmax(sim_matrix, axis=1).tolist()
            logger.info("Predictions made using label embeddings similarity.")
            return predictions
        except Exception as e:
            logger.error("Failed to predict with label embeddings: %s", e)
            raise


# def main():
#     # Example usage of TextClassifier

#     # Define configurations
#     model_config = ModelConfig(
#         model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
#         task="sentiment-analysis",
#         device=0  # Use GPU 0
#     )

#     dataset_config = DatasetConfig(
#         name="My_dataset"
#     )

#     # Initialize classifier
#     classifier = TextClassifier(
#         model_config=model_config,
#         dataset_config=dataset_config
#     )

#     # Load data
#     classifier.load_data()

#     # Initialize pipeline
#     classifier.initialize_pipeline()

#     # Run inference
#     y_pred = classifier.run_inference(split="test", text_column="text")

#     # Evaluate performance
#     classifier.evaluate_performance(
#         split="test",
#         y_pred=y_pred,
#         label_column="label",
#         target_names=["Negative Review", "Positive Review"]
#     )

#     # Text Classification with Embeddings and Logistic Regression
#     train_embeddings, train_labels = classifier.encode_embeddings(split="train", text_column="text")
#     test_embeddings, _ = classifier.encode_embeddings(split="test", text_column="text")  # Assuming labels are same
#     classifier.train_logistic_regression(train_embeddings, train_labels)
#     lr_predictions = classifier.predict_with_classifier(test_embeddings)
#     classifier.evaluate_performance(
#         split="test",
#         y_pred=lr_predictions,
#         label_column="label",
#         target_names=["Negative Review", "Positive Review"]
#     )

#     # Text Classification with Averaged Embeddings and Cosine Similarity
#     averaged_embeddings = classifier.average_embeddings(train_embeddings, train_labels)
#     cosine_predictions = classifier.predict_with_cosine_similarity(test_embeddings, averaged_embeddings)
#     classifier.evaluate_performance(
#         split="test",
#         y_pred=cosine_predictions,
#         label_column="label",
#         target_names=["Negative Review", "Positive Review"]
#     )

#     # Text Classification with Label Embeddings
#     label_embeddings = classifier.create_label_embeddings(["A negative review", "A positive review"])
#     label_emb_predictions = classifier.predict_with_label_embeddings(test_embeddings, label_embeddings)
#     classifier.evaluate_performance(
#         split="test",
#         y_pred=label_emb_predictions,
#         label_column="label",
#         target_names=["Negative Review", "Positive Review"]
#     )

#     # Text Generation for Sentiment Analysis using T5
#     t5_model_config = ModelConfig(
#         model_path="google/flan-t5-small",
#         task="text2text-generation",
#         device=0  # Use GPU 0
#     )

#     t5_classifier = TextClassifier(
#         model_config=t5_model_config,
#         dataset_config=dataset_config,
#         prompt="Is the following sentence positive or negative? "
#     )

#     t5_classifier.load_data()
#     t5_classifier.initialize_pipeline()
#     t5_predictions = t5_classifier.run_inference(
#         split="test",
#         text_column="text",
#         generation_column="t5"
#     )
#     t5_classifier.evaluate_performance(
#         split="test",
#         y_pred=t5_predictions,
#         label_column="label",
#         target_names=["Negative Review", "Positive Review"]
#     )


# if __name__ == "__main__":
#     main()