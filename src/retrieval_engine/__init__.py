"""retrieval-engine: a production retrieval-augmented generation service.

The package is organised by pipeline stage: :mod:`ingest` turns files into chunks,
:mod:`embed` turns chunks into vectors, :mod:`store` persists them, :mod:`retrieve`
finds candidates, :mod:`generate` answers from them, :mod:`guardrails` checks those
answers, :mod:`api` exposes the whole thing, and :mod:`eval` measures it.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
