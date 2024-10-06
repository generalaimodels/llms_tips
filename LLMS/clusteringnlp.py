import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset
from hdbscan import HDBSCAN
from plotly import express as px
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading of datasets with flexibility for different sources and configurations.
    """

    def __init__(
        self,
        dataset_name: str,
        split: Optional[List[str]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the DataLoader.

        :param dataset_name: Name of the dataset to load.
        :param split: List of splits to load (e.g., ['train', 'validation', 'test']).
        :param column_mapping: Mapping of standardized column names to actual dataset column names.
                               e.g., {'text': 'abstract', 'label': 'category'}
        """
        self.dataset_name = dataset_name
        self.split = split or ['train', 'validation', 'test']
        self.column_mapping = column_mapping or {'text': 'text', 'label': 'label'}
        self.data: Optional[DatasetDict] = None

    def load(self) -> DatasetDict:
        """
        Loads the dataset and applies column mappings.

        :return: Loaded and mapped DatasetDict.
        """
        try:
            logger.info(f"Loading dataset '{self.dataset_name}' with splits {self.split}")
            self.data = load_dataset(self.dataset_name, split=self.split)
            logger.info("Dataset loaded successfully.")

            # Apply column mapping if necessary
            if self.column_mapping != {'text': 'text', 'label': 'label'}:
                logger.info("Applying column mappings.")
                for split in self.split:
                    if split in self.data:
                        self.data[split] = self.data[split].map(
                            lambda example: {
                                'text': example[self.column_mapping['text']],
                                'label': example[self.column_mapping['label']],
                            }
                        )
            return self.data
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise


class Embedder:
    """
    Generates embeddings for text data using SentenceTransformer.
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes the Embedder.

        :param model_name: Pretrained SentenceTransformer model name.
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None

    def load_model(self) -> SentenceTransformer:
        """
        Loads the SentenceTransformer model.

        :return: Loaded SentenceTransformer model.
        """
        try:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'.")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully.")
            return self.model
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Generates embeddings for a list of texts.

        :param texts: List of text strings to encode.
        :param show_progress_bar: Whether to display a progress bar.
        :return: Numpy array of embeddings.
        """
        if not self.model:
            self.load_model()
        try:
            logger.info("Encoding texts into embeddings.")
            embeddings = self.model.encode(texts, show_progress_bar=show_progress_bar)
            logger.info("Text encoding completed.")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise


class Reducer:
    """
    Reduces dimensionality of embeddings using UMAP.
    """

    def __init__(
        self,
        n_components: int = 5,
        min_dist: float = 0.0,
        metric: str = 'cosine',
        random_state: int = 42,
    ):
        """
        Initializes the Reducer.

        :param n_components: Target number of dimensions.
        :param min_dist: Minimum distance parameter for UMAP.
        :param metric: Metric to use for UMAP.
        :param random_state: Random seed for reproducibility.
        """
        self.umap_model = UMAP(
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        logger.info(f"UMAP initialized with n_components={n_components}, min_dist={min_dist}, metric={metric}")

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fits the UMAP model and transforms the embeddings.

        :param embeddings: Numpy array of embeddings.
        :return: Reduced embeddings.
        """
        try:
            logger.info("Fitting UMAP and transforming embeddings.")
            reduced = self.umap_model.fit_transform(embeddings)
            logger.info("Dimensionality reduction completed.")
            return reduced
        except Exception as e:
            logger.error(f"Error during dimensionality reduction: {e}")
            raise


class Clusterer:
    """
    Performs clustering on reduced embeddings using HDBSCAN.
    """

    def __init__(
        self,
        min_cluster_size: int = 50,
        metric: str = 'euclidean',
        cluster_selection_method: str = 'eom',
    ):
        """
        Initializes the Clusterer.

        :param min_cluster_size: Minimum size of clusters.
        :param metric: Metric to use for clustering.
        :param cluster_selection_method: Method to select clusters.
        """
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method=cluster_selection_method
        )
        logger.info(f"HDBSCAN initialized with min_cluster_size={min_cluster_size}, metric={metric}, "
                    f"cluster_selection_method={cluster_selection_method}")

    def fit(self, reduced_embeddings: np.ndarray) -> np.ndarray:
        """
        Fits the HDBSCAN model and predicts cluster labels.

        :param reduced_embeddings: Numpy array of reduced embeddings.
        :return: Cluster labels.
        """
        try:
            logger.info("Fitting HDBSCAN and predicting cluster labels.")
            clusters = self.hdbscan_model.fit_predict(reduced_embeddings)
            num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            logger.info(f"Clustering completed with {num_clusters} clusters found.")
            return clusters
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise


class Visualizer:
    """
    Creates interactive visualizations using Plotly.
    """

    @staticmethod
    def plot_clusters(
        df: pd.DataFrame,
        outlier_label: int = -1,
        point_size: int = 5
    ) -> Any:
        """
        Plots clusters and outliers using Plotly.

        :param df: DataFrame containing 'x', 'y', and 'cluster' columns.
        :param outlier_label: Label used to identify outliers.
        :param point_size: Size of the scatter plot points.
        :return: Plotly figure object.
        """
        try:
            logger.info("Creating cluster visualization plot.")
            clusters_df = df[df['cluster'] != str(outlier_label)]
            outliers_df = df[df['cluster'] == str(outlier_label)]

            fig = px.scatter(
                clusters_df,
                x='x',
                y='y',
                color='cluster',
                labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
                title='Clusters Visualization',
                hover_data=['title'],
                width=800,
                height=600
            )

            # Add outliers
            fig.add_trace(
                px.scatter(
                    outliers_df,
                    x='x',
                    y='y',
                    marker=dict(color='grey', size=point_size, opacity=0.5),
                    name='Outliers'
                ).data[0]
            )

            fig.update_layout(showlegend=True)
            logger.info("Cluster visualization plot created.")
            return fig
        except Exception as e:
            logger.error(f"Error during plotting clusters: {e}")
            raise

    @staticmethod
    def display_plot(fig: Any) -> None:
        """
        Displays the Plotly figure.

        :param fig: Plotly figure object.
        """
        try:
            fig.show()
        except Exception as e:
            logger.error(f"Error displaying plot: {e}")
            raise


class TopicModeler:
    """
    Performs topic modeling using BERTopic.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        umap_model: UMAP,
        hdbscan_model: HDBSCAN,
        verbose: bool = True
    ):
        """
        Initializes the TopicModeler.

        :param embedding_model: Pretrained SentenceTransformer model.
        :param umap_model: Trained UMAP model.
        :param hdbscan_model: Trained HDBSCAN model.
        :param verbose: Whether to enable verbose output.
        """
        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=verbose
        )
        logger.info("BERTopic model initialized.")

    def fit(self, documents: List[str], embeddings: np.ndarray) -> None:
        """
        Fits the BERTopic model on the provided documents and embeddings.

        :param documents: List of documents to model.
        :param embeddings: Numpy array of embeddings.
        """
        try:
            logger.info("Fitting BERTopic model.")
            self.topic_model.fit(documents, embeddings)
            logger.info("BERTopic model fitted successfully.")
        except Exception as e:
            logger.error(f"Error fitting BERTopic model: {e}")
            raise

    def get_topic_info(self) -> pd.DataFrame:
        """
        Retrieves the topic information.

        :return: DataFrame containing topic information.
        """
        try:
            return self.topic_model.get_topic_info()
        except Exception as e:
            logger.error(f"Error retrieving topic info: {e}")
            raise

    def get_topic(self, topic: int) -> List[Tuple[str, float]]:
        """
        Retrieves the keywords and their weights for a specific topic.

        :param topic: Topic number.
        :return: List of tuples containing keywords and their weights.
        """
        try:
            return self.topic_model.get_topic(topic)
        except Exception as e:
            logger.error(f"Error retrieving topic {topic}: {e}")
            raise

    def visualize_documents(
        self,
        texts: List[str],
        reduced_embeddings: np.ndarray,
        width: int = 1200,
        hide_annotations: bool = True
    ) -> Any:
        """
        Visualizes documents in the topic model.

        :param texts: List of document titles or identifiers.
        :param reduced_embeddings: Numpy array of reduced embeddings.
        :param width: Width of the visualization.
        :param hide_annotations: Whether to hide annotations.
        :return: Plotly figure object.
        """
        try:
            logger.info("Creating documents visualization.")
            fig = self.topic_model.visualize_documents(
                texts,
                reduced_embeddings=reduced_embeddings,
                width=width,
                hide_annotations=hide_annotations
            )
            logger.info("Documents visualization created.")
            return fig
        except Exception as e:
            logger.error(f"Error visualizing documents: {e}")
            raise

    def visualize_barchart(self) -> Any:
        """
        Visualizes a bar chart of topic keywords.

        :return: Plotly figure object.
        """
        try:
            logger.info("Creating topic keywords bar chart.")
            fig = self.topic_model.visualize_barchart()
            logger.info("Barchart created.")
            return fig
        except Exception as e:
            logger.error(f"Error creating barchart: {e}")
            raise

    def visualize_heatmap(self, n_clusters: int = 30) -> Any:
        """
        Visualizes relationships between topics as a heatmap.

        :param n_clusters: Number of clusters to group topics.
        :return: Plotly figure object.
        """
        try:
            logger.info("Creating heatmap visualization.")
            fig = self.topic_model.visualize_heatmap(n_clusters=n_clusters)
            logger.info("Heatmap created.")
            return fig
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            raise

    def visualize_hierarchy(self) -> Any:
        """
        Visualizes the hierarchical structure of topics.

        :return: Plotly figure object.
        """
        try:
            logger.info("Creating hierarchy visualization.")
            fig = self.topic_model.visualize_hierarchy()
            logger.info("Hierarchy visualization created.")
            return fig
        except Exception as e:
            logger.error(f"Error creating hierarchy visualization: {e}")
            raise


class Pipeline:
    """
    Orchestrates the entire data processing, embedding, clustering, and visualization workflow.
    """

    def __init__(
        self,
        dataset_name: str,
        column_mapping: Optional[Dict[str, str]] = None,
        embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        umap_kwargs: Optional[Dict[str, Any]] = None,
        hdbscan_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the Pipeline.

        :param dataset_name: Name of the dataset to load.
        :param column_mapping: Mapping of standardized column names to actual dataset column names.
        :param embedding_model_name: Pretrained SentenceTransformer model name.
        :param umap_kwargs: Keyword arguments for UMAP.
        :param hdbscan_kwargs: Keyword arguments for HDBSCAN.
        """
        self.data_loader = DataLoader(dataset_name, column_mapping=column_mapping)
        self.embedder = Embedder(model_name=embedding_model_name)
        self.umap_kwargs = umap_kwargs or {
            'n_components': 2,
            'min_dist': 0.0,
            'metric': 'cosine',
            'random_state': 42
        }
        self.reducer = Reducer(**self.umap_kwargs)
        self.hdbscan_kwargs = hdbscan_kwargs or {
            'min_cluster_size': 50,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom'
        }
        self.clusterer = Clusterer(**self.hdbscan_kwargs)
        self.topic_modeler: Optional[TopicModeler] = None
        self.df: Optional[pd.DataFrame] = None

    def run(self) -> None:
        """
        Executes the entire pipeline.
        """
        try:
            # Load data
            data = self.data_loader.load()
            train_data = data['train']
            abstracts = train_data[self.data_loader.column_mapping['text']]
            labels = train_data[self.data_loader.column_mapping['label']]
            titles = [f"Document {i}" for i in range(len(abstracts))]  # Placeholder for titles

            # Embed texts
            embeddings = self.embedder.encode(abstracts, show_progress_bar=True)

            # Reduce dimensionality
            reduced_embeddings = self.reducer.fit_transform(embeddings)

            # Cluster embeddings
            clusters = self.clusterer.fit(reduced_embeddings)

            # Create DataFrame for visualization
            self.df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
            self.df['title'] = titles
            self.df['cluster'] = clusters.astype(str)

            # Visualize clusters
            visualizer = Visualizer()
            fig = visualizer.plot_clusters(self.df)
            visualizer.display_plot(fig)

            # Initialize and fit BERTopic
            self.topic_modeler = TopicModeler(
                embedding_model=self.embedder.model,
                umap_model=self.reducer.umap_model,
                hdbscan_model=self.clusterer.hdbscan_model,
                verbose=True
            )
            self.topic_modeler.fit(abstracts, embeddings)

            # Display topic information
            topic_info = self.topic_modeler.get_topic_info()
            logger.info(f"Topic Information:\n{topic_info}")

            # Visualize topics
            doc_vis = self.topic_modeler.visualize_documents(
                titles,
                reduced_embeddings=self.reducer.umap_model.transform(embeddings)
            )
            doc_vis.show()

            # Additional visualizations
            barchart = self.topic_modeler.visualize_barchart()
            barchart.show()

            heatmap = self.topic_modeler.visualize_heatmap()
            heatmap.show()

            hierarchy = self.topic_modeler.visualize_hierarchy()
            hierarchy.show()

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


def main():
    """
    Entry point for the module.
    """
    # Configuration parameters
    dataset_name = "My_dataset"  # Replace with your actual dataset name
    column_mapping = {'text': 'text', 'label': 'label'}  # Adjust if your dataset has different column names
    embedding_model_name = 'thenlper/gte-small'  # Replace with your desired embedding model

    # Optional UMAP and HDBSCAN parameters
    umap_kwargs = {
        'n_components': 2,
        'min_dist': 0.0,
        'metric': 'cosine',
        'random_state': 42
    }
    hdbscan_kwargs = {
        'min_cluster_size': 50,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom'
    }

    # Initialize and run the pipeline
    pipeline = Pipeline(
        dataset_name=dataset_name,
        column_mapping=column_mapping,
        embedding_model_name=embedding_model_name,
        umap_kwargs=umap_kwargs,
        hdbscan_kwargs=hdbscan_kwargs
    )
    pipeline.run()


if __name__ == "__main__":
    main()