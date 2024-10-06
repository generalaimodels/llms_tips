#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Topic Modeling Pipeline

This module provides a reusable and scalable pipeline for topic modeling using
state-of-the-art techniques. It includes data loading, embedding generation,
dimensionality reduction, clustering, and visualization functionalities.

Usage:
    from topic_modeling_pipeline import TopicModelingPipeline

    pipeline = TopicModelingPipeline(
        dataset_name="your_dataset_name",
        text_column="text",
        embedding_model_name="thenlper/gte-small"
    )
    pipeline.run()
"""

import logging
from typing import Any, List, Optional

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from datasets import DatasetDict, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.exceptions import NotFittedError
from umap import UMAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class TopicModelingPipeline:
    """
    A pipeline for topic modeling that includes data loading, embeddings generation,
    dimensionality reduction, clustering, and visualization.
    """

    def __init__(
        self,
        dataset_name: str,
        text_column: str = "text",
        embedding_model_name: str = "thenlper/gte-small",
        umap_n_components: int = 5,
        hdbscan_min_cluster_size: int = 50,
        random_state: Optional[int] = 42
    ) -> None:
        """
        Initializes the TopicModelingPipeline with specified parameters.

        Args:
            dataset_name (str): The name of the dataset to load.
            text_column (str): The name of the text column in the dataset.
            embedding_model_name (str): The name of the embedding model to use.
            umap_n_components (int): Number of components for UMAP.
            hdbscan_min_cluster_size (int): Minimum cluster size for HDBSCAN.
            random_state (Optional[int]): Random state for reproducibility.
        """
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.embedding_model_name = embedding_model_name
        self.umap_n_components = umap_n_components
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.random_state = random_state

        self.data: Optional[DatasetDict] = None
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.reduced_embeddings: Optional[np.ndarray] = None
        self.clusters: Optional[np.ndarray] = None
        self.df: Optional[pd.DataFrame] = None
        self.topic_model: Optional[BERTopic] = None

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.umap_model = UMAP(
            n_components=self.umap_n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=self.random_state
        )
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )

    def load_data(self) -> None:
        """Loads the dataset specified in the initialization."""
        try:
            logging.info(f"Loading dataset: {self.dataset_name}")
            self.data = load_dataset(self.dataset_name)
            # Concatenate datasets
            for split in self.data.keys():
                self.texts.extend(self.data[split][self.text_column])
            logging.info("Dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise

    def generate_embeddings(self) -> None:
        """Generates embeddings for the loaded texts."""
        if not self.texts:
            raise ValueError("No texts available for embedding generation.")
        try:
            logging.info("Generating embeddings...")
            self.embeddings = self.embedding_model.encode(
                self.texts,
                show_progress_bar=True
            )
            logging.info("Embeddings generated successfully.")
        except Exception as e:
            logging.error(f"Failed to generate embeddings: {e}")
            raise

    def reduce_dimensions(self) -> None:
        """Reduces the dimensionality of embeddings using UMAP."""
        if self.embeddings is None:
            raise ValueError("Embeddings have not been generated.")
        try:
            logging.info("Reducing dimensions using UMAP...")
            self.reduced_embeddings = self.umap_model.fit_transform(self.embeddings)
            logging.info("Dimensionality reduction completed.")
        except Exception as e:
            logging.error(f"Failed to reduce dimensions: {e}")
            raise

    def cluster_embeddings(self) -> None:
        """Clusters reduced embeddings using HDBSCAN."""
        if self.reduced_embeddings is None:
            raise ValueError("Reduced embeddings are not available.")
        try:
            logging.info("Clustering embeddings using HDBSCAN...")
            self.hdbscan_model.fit(self.reduced_embeddings)
            self.clusters = self.hdbscan_model.labels_
            num_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
            logging.info(f"Number of clusters found: {num_clusters}")
        except Exception as e:
            logging.error(f"Failed to cluster embeddings: {e}")
            raise

    def create_dataframe(self) -> None:
        """Creates a DataFrame with reduced embeddings and cluster assignments."""
        if self.reduced_embeddings is None or self.clusters is None:
            raise ValueError("Required data for DataFrame is missing.")
        try:
            logging.info("Creating DataFrame for visualization...")
            self.df = pd.DataFrame(
                self.reduced_embeddings,
                columns=["x", "y"]
            )
            self.df["text"] = self.texts
            self.df["cluster"] = self.clusters.astype(str)
            logging.info("DataFrame created successfully.")
        except Exception as e:
            logging.error(f"Failed to create DataFrame: {e}")
            raise

    def visualize_clusters(self) -> None:
        """Visualizes clusters using Plotly scatter plot."""
        if self.df is None:
            raise ValueError("DataFrame is not available for visualization.")
        try:
            logging.info("Visualizing clusters...")
            fig = px.scatter(
                self.df,
                x="x",
                y="y",
                color="cluster",
                hover_data=["text"],
                title="Cluster Visualization"
            )
            fig.show()
            logging.info("Visualization completed.")
        except Exception as e:
            logging.error(f"Failed to visualize clusters: {e}")
            raise

    def train_topic_model(self) -> None:
        """Trains a BERTopic model on the texts and embeddings."""
        if not self.texts or self.embeddings is None:
            raise ValueError("Texts or embeddings are not available for training.")
        try:
            logging.info("Training BERTopic model...")
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                verbose=True
            )
            self.topic_model.fit(self.texts, self.embeddings)
            logging.info("BERTopic model trained successfully.")
        except Exception as e:
            logging.error(f"Failed to train BERTopic model: {e}")
            raise

    def visualize_topics(self) -> None:
        """Visualizes topics using BERTopic's visualization tools."""
        if self.topic_model is None:
            raise ValueError("BERTopic model has not been trained.")
        try:
            logging.info("Visualizing topics...")
            # Visualize Bar Chart
            bar_chart = self.topic_model.visualize_barchart()
            bar_chart.show()

            # Visualize Hierarchical Structure
            hierarchy = self.topic_model.visualize_hierarchy()
            hierarchy.show()

            # Visualize Heatmap
            heatmap = self.topic_model.visualize_heatmap()
            heatmap.show()

            logging.info("Topic visualization completed.")
        except Exception as e:
            logging.error(f"Failed to visualize topics: {e}")
            raise

    def run(self) -> None:
        """Executes the full pipeline."""
        try:
            self.load_data()
            self.generate_embeddings()
            self.reduce_dimensions()
            self.cluster_embeddings()
            self.create_dataframe()
            self.visualize_clusters()
            self.train_topic_model()
            self.visualize_topics()
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    # Example usage of the pipeline
    pipeline = TopicModelingPipeline(
        dataset_name="your_dataset_name",
        text_column="text",
        embedding_model_name="thenlper/gte-small"
    )
    pipeline.run()