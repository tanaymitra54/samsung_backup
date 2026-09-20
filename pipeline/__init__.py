"""Public pipeline exports. Heavy modules load only when requested."""

from __future__ import annotations

__all__ = [
    "DiverseSampler",
    "ReasonVerifier",
    "QUBOBuilder",
    "SimulatedAnnealingSolver",
    "QuantumSolver",
    "make_solver",
    "InferencePipeline",
    "HyperparameterQUBO",
    "PRISMPipeline",
    "check_flow",
    "run_one_query",
]


def __getattr__(name: str):
    if name in ("DiverseSampler",):
        from .sampling import DiverseSampler

        return DiverseSampler
    if name in ("ReasonVerifier",):
        from .verifier import ReasonVerifier

        return ReasonVerifier
    if name in ("QUBOBuilder",):
        from .qubo_builder import QUBOBuilder

        return QUBOBuilder
    if name in ("SimulatedAnnealingSolver", "QuantumSolver", "make_solver"):
        from . import solver

        return getattr(solver, name)
    if name in ("InferencePipeline",):
        from .inference import InferencePipeline

        return InferencePipeline
    if name in ("HyperparameterQUBO",):
        from .hyperparam_qubo import HyperparameterQUBO

        return HyperparameterQUBO
    if name in ("PRISMPipeline", "check_flow", "run_one_query"):
        from . import orchestrator

        return getattr(orchestrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
