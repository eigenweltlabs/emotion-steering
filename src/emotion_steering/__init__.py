"""emotion-steering: extract and serve activation-based emotion steering vectors (Konen et al. 2024)."""

__version__ = "0.1.0"

from .vectors import VectorBundle, load_bundle, save_bundle

__all__ = ["VectorBundle", "load_bundle", "save_bundle", "__version__"]
