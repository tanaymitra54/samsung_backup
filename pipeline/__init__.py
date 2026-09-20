"""Public pipeline exports. Heavy modules load only when requested."""

from __future__ import annotations

__all__ = [
    "DiverseSampler",
    "ReasonVerifier",
    "QUBOBuilder",
    "SimulatedAnnealingSolver",
    "InferencePipeline",
    "HyperparameterQUBO",
    "run_reasoning_pipeline",
]


def __getattr__(name: str):
    if name == "DiverseSampler":
        from .sampling import DiverseSampler

        return DiverseSampler
    if name == "ReasonVerifier":
        from .verifier import ReasonVerifier

        return ReasonVerifier
    if name == "QUBOBuilder":
        from .qubo_builder import QUBOBuilder

        return QUBOBuilder
    if name in ("SimulatedAnnealingSolver", "QuantumSolver", "make_solver"):
        from . import solver

        return getattr(solver, name)
    if name == "InferencePipeline":
        from .inference import InferencePipeline

        return InferencePipeline
    if name == "HyperparameterQUBO":
        from .hyperparam_qubo import HyperparameterQUBO

        return HyperparameterQUBO
    if name == "run_reasoning_pipeline":
        from .reasoning import run_reasoning_pipeline

        return run_reasoning_pipeline
    if name in ("PRISMPipeline", "check_flow", "run_one_query"):
        from . import orchestrator

        return getattr(orchestrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
