from importlib.metadata import version, PackageNotFoundError

from tsf_embedding import TransformerEmbedding
from t5_inference import T5Inference

try:
    __version__: str = version("rag_lib")
except PackageNotFoundError:
    # Package is not installed (e.g., running locally during development)
    __version__ = "0.0.0.dev0"

__all__ = ["TransformerEmbedding", "T5Inference"]
