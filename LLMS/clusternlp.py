import logging
from typing import Optional, Union, List, Dict, Any

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px
from datasets import load_dataset

class TextClusterer:
    def __init__(self,
                 embedding_model_name: str = 'thenlper/gte-small',
                 n_components: int = 5,
                 min_cluster_size: int = 50,
                 umap_random_state: Optional[int] = 42,
                 umap_metric: str = 'cosine',
                 hdbscan_metric: str = 'euclidean',
                 hdbscan_cluster_selection_method: str = 'eom',
                 ):
        """
        Initialize the TextClusterer with specified models and parameters.
        
        :param embedding_model_name: The name of the SentenceTransformer model to use for embeddings.
        :param n_components: The number of dimensions to reduce embeddings to with UMAP.
        :param min_cluster_size: The minimum size of clusters for HDBSCAN.
        :param umap_random_state: Random state for UMAP dimensionality reduction.
        :param umap_metric: Metric used by UMAP.
        :param hdbscan_metric: Metric used by HDBSCAN.
        :param hdbscan_cluster_selection_method: Cluster selection method for HDBSCAN.
        """
        self.embedding_model_name = embedding_model_name
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.umap_random_state = umap_random_state
        self.umap_metric = umap_metric
        self.hdbscan_metric = hdbscan_metric
        self.hdbscan_cluster_selection_method = hdbscan_cluster_selection_method

        self.embedding_model: Optional[Any] = None
        self.umap_model: Optional[UMAP] = None
        self.hdbscan_model: Optional[HDBSCAN] = None

        # Data containers
        self.embeddings: Optional[np.ndarray] = None
        self.reduced_embeddings: Optional[np.ndarray] = None
        self.clusters: Optional[np.ndarray] = None
        self.df: Optional[pd.DataFrame] = None  # DataFrame for visualization

    def load_dataset(self, dataset_name: str, split: Union[str, List[str]] = 'train', text_column: str = 'text') -> pd.DataFrame:
        """
        Load dataset using Huggingface load_dataset function.

        :param dataset_name: Name of the dataset to load.
        :param split: Split or list of splits to load.
        :param text_column: Name of the column containing the text data.
        :return: A pandas DataFrame containing the loaded data.
        """
        try:
            dataset = load_dataset(dataset_name, split=split)
            logging.info(f"Loaded dataset '{dataset_name}' with split '{split}'")
            if isinstance(dataset, dict):
                # If dataset is a DatasetDict (multiple splits)
                dataframes = {key: pd.DataFrame(dataset[key]) for key in dataset.keys()}
                df = pd.concat(dataframes.values(), ignore_index=True)
            else:
                # Single split
                df = pd.DataFrame(dataset)
            # Ensure text_column exists
            if text_column not in df.columns:
                raise KeyError(f"Column '{text_column}' not found in the dataset.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while loading dataset: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        :param texts: List of texts to generate embeddings for.
        :return: Numpy array of embeddings.
        """
        try:
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            self.embeddings = embeddings
            logging.info("Generated embeddings.")
            return embeddings
        except Exception as e:
            logging.error(f"An error occurred while generating embeddings: {e}")
            raise

    def reduce_dimensions(self, embeddings: Optional[np.ndarray] = None, n_components: Optional[int] = None) -> np.ndarray:
        """
        Reduce the dimensions of embeddings using UMAP.

        :param embeddings: Numpy array of embeddings to reduce. If None, uses self.embeddings.
        :param n_components: Number of components to reduce to. If None, uses self.n_components.
        :return: Reduced embeddings.
        """
        try:
            if embeddings is None:
                embeddings = self.embeddings
            if embeddings is None:
                raise ValueError("Embeddings have not been generated. Call generate_embeddings() first.")

            if n_components is None:
                n_components = self.n_components

            self.umap_model = UMAP(n_components=n_components, min_dist=0.0, metric=self.umap_metric, random_state=self.umap_random_state)
            reduced_embeddings = self.umap_model.fit_transform(embeddings)
            self.reduced_embeddings = reduced_embeddings
            logging.info(f"Reduced embeddings to {n_components} dimensions.")
            return reduced_embeddings
        except Exception as e:
            logging.error(f"An error occurred during dimension reduction: {e}")
            raise

    def cluster_embeddings(self, embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN.

        :param embeddings: The embeddings to cluster. If None, uses reduced embeddings if available, else embeddings.
        :return: Numpy array of cluster labels.
        """
        try:
            if embeddings is None:
                embeddings = self.reduced_embeddings if self.reduced_embeddings is not None else self.embeddings
            if embeddings is None:
                raise ValueError("Embeddings have not been generated. Call generate_embeddings() first.")

            self.hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric=self.hdbscan_metric,
                cluster_selection_method=self.hdbscan_cluster_selection_method
            ).fit(embeddings)
            self.clusters = self.hdbscan_model.labels_
            logging.info("Clustered embeddings.")
            return self.clusters
        except Exception as e:
            logging.error(f"An error occurred during clustering: {e}")
            raise

    def prepare_dataframe(self, texts: List[str], extra_columns: Optional[Dict[str, List[Any]]] = None) -> pd.DataFrame:
        """
        Prepare pandas DataFrame for visualization.

        :param texts: List of texts corresponding to embeddings.
        :param extra_columns: Dictionary of extra columns to add to the DataFrame.
        :return: pandas DataFrame with necessary columns for visualization.
        """
        try:
            if self.reduced_embeddings is None:
                raise ValueError("Reduced embeddings are not available. Call reduce_dimensions() first.")

            df = pd.DataFrame(self.reduced_embeddings, columns=[f"x_{i}" for i in range(self.reduced_embeddings.shape[1])])
            df['text'] = texts
            df['cluster'] = [str(c) for c in self.clusters] if self.clusters is not None else None

            if extra_columns:
                for col_name, col_values in extra_columns.items():
                    df[col_name] = col_values

            self.df = df
            logging.info("DataFrame prepared for visualization.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while preparing DataFrame: {e}")
            raise

    def visualize_clusters(self, dimensions: int = 2, title: str = "Cluster Visualization") -> None:
        """
        Visualize the clusters using Plotly.

        :param dimensions: The number of dimensions to plot. Currently supports 2 or 3.
        :param title: Title of the plot.
        """
        try:
            if self.df is None:
                raise ValueError("DataFrame not prepared. Call prepare_dataframe() first.")

            if dimensions == 2:
                fig = px.scatter(
                    self.df, x='x_0', y='x_1', color='cluster', hover_data=['text'],
                    title=title
                )
            elif dimensions == 3:
                if 'x_2' not in self.df.columns:
                    raise ValueError("Reduced embeddings do not have 3 dimensions. Reduce embeddings to 3 dimensions to plot in 3D.")
                fig = px.scatter_3d(
                    self.df, x='x_0', y='x_1', z='x_2', color='cluster', hover_data=['text'],
                    title=title
                )
            else:
                raise ValueError("Only 2D and 3D visualizations are supported.")

            fig.show()
        except Exception as e:
            logging.error(f"An error occurred during visualization: {e}")
            raise

    # Additional methods can be added for further analysis and visualization as necessary.

# # If being run as a script, we might include some example usage
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     clusterer = TextClusterer()

#     # Example usage:
#     # 1. Load data
#     try:
#         data_df = clusterer.load_dataset('YourDatasetName', split='train', text_column='your_text_column')
#         texts = data_df['your_text_column'].tolist()
#     except Exception as e:
#         logging.error(f"Failed to load dataset: {e}")
#         exit(1)

#     # 2. Generate embeddings
#     clusterer.generate_embeddings(texts)

#     # 3. Reduce dimensions
#     clusterer.reduce_dimensions()

#     # 4. Cluster embeddings
#     clusterer.cluster_embeddings()

#     # 5. Prepare dataframe for visualization
#     clusterer.prepare_dataframe(texts)

#     # 6. Visualize clusters
#     clusterer.visualize_clusters()