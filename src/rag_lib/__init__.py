from importlib.metadata import version, PackageNotFoundError

from rag_lib.tsf_embedding import TSF_Embedding
from rag_lib.causal_inference import CausalInference

try:
    __version__: str = version("rag_lib")
except PackageNotFoundError:
    # Package is not installed (e.g., running locally during development)
    __version__ = "0.0.0.dev0"

__all__ = ["TSF_Embedding", "CausalInference"]
