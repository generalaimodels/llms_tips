# sentiment_analysis_pipeline.py

import logging
from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm


@dataclass
class SentimentAnalysisConfig:
    dataset_name: str
    text_column: str = 'text'
    label_column: str = 'label'
    model_name: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    embedding_model_name: str = 'sentence-transformers/all-mpnet-base-v2'
    device: Union[str, int] = 'cuda:0'
    prompt_template: str = 'Is the following sentence positive or negative? '
    few_shot_labels: List[str] = None
    task_type: str = 'classification'  # Options: classification, zero_shot, few_shot, text_generation


class SentimentAnalysisPipeline:
    def __init__(self, config: SentimentAnalysisConfig):
        self.config = config
        self.data = DatasetDict()
        self.pipeline = None
        self.embedding_model = None
        self.classifier = None
        self.label_embeddings = None
        self.logger = logging.getLogger(__name__)
        self._setup_logger()
        self._load_data()
        self._load_models()

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        self.logger.info("Logger is set up.")

    def _load_data(self):
        try:
            self.data = load_dataset(self.config.dataset_name)
            self.logger.info(f"Dataset '{self.config.dataset_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def _load_models(self):
        try:
            self.pipeline = pipeline(
                model=self.config.model_name,
                tokenizer=self.config.model_name,
                return_all_scores=True,
                device=self.config.device
            )
            self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
            self.logger.info("Models loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def run_classification(self):
        y_pred = []
        key_dataset = KeyDataset(self.data['test'], self.config.text_column)
        self.logger.info("Starting inference...")
        for output in tqdm(self.pipeline(key_dataset), total=len(self.data['test'])):
            scores = {item['label']: item['score'] for item in output}
            assignment = np.argmax([scores.get('LABEL_0', 0), scores.get('LABEL_2', 0)])
            y_pred.append(assignment)
        self.evaluate_performance(self.data['test'][self.config.label_column], y_pred)

    def run_embedding_classification(self):
        train_texts = self.data['train'][self.config.text_column]
        test_texts = self.data['test'][self.config.text_column]
        train_labels = self.data['train'][self.config.label_column]
        test_labels = self.data['test'][self.config.label_column]

        self.logger.info("Encoding texts into embeddings...")
        train_embeddings = self.embedding_model.encode(train_texts, show_progress_bar=True)
        test_embeddings = self.embedding_model.encode(test_texts, show_progress_bar=True)

        self.logger.info("Training classifier...")
        self.classifier = LogisticRegression(random_state=42)
        self.classifier.fit(train_embeddings, train_labels)

        self.logger.info("Predicting on test data...")
        y_pred = self.classifier.predict(test_embeddings)
        self.evaluate_performance(test_labels, y_pred)

    def run_zero_shot_classification(self):
        train_texts = self.data['train'][self.config.text_column]
        test_texts = self.data['test'][self.config.text_column]
        train_labels = self.data['train'][self.config.label_column]
        test_labels = self.data['test'][self.config.label_column]

        self.logger.info("Encoding texts into embeddings...")
        train_embeddings = self.embedding_model.encode(train_texts, show_progress_bar=True)
        test_embeddings = self.embedding_model.encode(test_texts, show_progress_bar=True)

        self.logger.info("Calculating average embeddings per label...")
        df = pd.DataFrame(train_embeddings)
        df[self.config.label_column] = train_labels
        averaged_embeddings = df.groupby(self.config.label_column).mean().values

        self.logger.info("Computing cosine similarity...")
        sim_matrix = cosine_similarity(test_embeddings, averaged_embeddings)
        y_pred = np.argmax(sim_matrix, axis=1)
        self.evaluate_performance(test_labels, y_pred)

    def run_few_shot_classification(self):
        if not self.config.few_shot_labels:
            self.config.few_shot_labels = ["A negative review", "A positive review"]

        test_texts = self.data['test'][self.config.text_column]
        test_labels = self.data['test'][self.config.label_column]

        self.logger.info("Encoding label descriptions into embeddings...")
        self.label_embeddings = self.embedding_model.encode(self.config.few_shot_labels)

        self.logger.info("Encoding test texts into embeddings...")
        test_embeddings = self.embedding_model.encode(test_texts, show_progress_bar=True)

        self.logger.info("Computing cosine similarity...")
        sim_matrix = cosine_similarity(test_embeddings, self.label_embeddings)
        y_pred = np.argmax(sim_matrix, axis=1)
        self.evaluate_performance(test_labels, y_pred)

    def run_text_generation(self):
        self.logger.info("Loading text generation model...")
        generation_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=self.config.device
        )

        self.logger.info("Preparing data with prompts...")
        prompt_column = 'prompted_text'
        self.data = self.data.map(lambda example: {
            prompt_column: self.config.prompt_template + example[self.config.text_column]
        })

        self.logger.info("Generating responses...")
        y_pred = []
        key_dataset = KeyDataset(self.data['test'], prompt_column)
        for output in tqdm(generation_pipeline(key_dataset), total=len(self.data['test'])):
            generated_text = output[0]["generated_text"].strip().lower()
            label = 0 if "negative" in generated_text else 1
            y_pred.append(label)

        self.evaluate_performance(self.data['test'][self.config.label_column], y_pred)

    @staticmethod
    def evaluate_performance(y_true: List[int], y_pred: List[int]):
        performance = classification_report(
            y_true, y_pred,
            target_names=["Negative Review", "Positive Review"]
        )
        print(performance)

    def run(self):
        task_methods = {
            'classification': self.run_classification,
            'embedding_classification': self.run_embedding_classification,
            'zero_shot': self.run_zero_shot_classification,
            'few_shot': self.run_few_shot_classification,
            'text_generation': self.run_text_generation
        }
        task_method = task_methods.get(self.config.task_type)
        if task_method:
            self.logger.info(f"Running task: {self.config.task_type}")
            task_method()
        else:
            self.logger.error(f"Task type '{self.config.task_type}' is not supported.")
            raise ValueError(f"Task type '{self.config.task_type}' is not supported.")


# if __name__ == "__main__":
#     config = SentimentAnalysisConfig(
#         dataset_name="my_dataset",
#         text_column="text",
#         label_column="label",
#         task_type="classification"
#     )
#     pipeline = SentimentAnalysisPipeline(config)
#     pipeline.run()