# test_opera.py:
from transformers import __version__ as tv
import inspect
from transformers.generation.utils import GenerationMixin
sig = inspect.signature(GenerationMixin.generate)
print("Transformers version:", tv)
print("Has opera_decoding param:", "opera_decoding" in sig.parameters)
