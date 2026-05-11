"""Synthetic NLP-like classification task generator.

Replace with a real loader (HuggingFace datasets, sklearn fetch_20newsgroups, etc.)
to point the harness at production data.
"""

from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

DEFAULT_DIM = 64
DEFAULT_CLASSES = 16
DEFAULT_SAMPLES = 4000


def make_dataset(n_samples=DEFAULT_SAMPLES, n_features=DEFAULT_DIM,
                 n_classes=DEFAULT_CLASSES, seed=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_features // 2, n_redundant=n_features // 4,
        n_classes=n_classes, n_clusters_per_class=2, class_sep=1.5,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.25, random_state=seed)


def feature_dim() -> int:
    return DEFAULT_DIM


def n_classes() -> int:
    return DEFAULT_CLASSES
