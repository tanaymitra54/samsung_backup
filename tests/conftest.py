"""Record the scope and environment of the completed test session."""
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    if not any(item.path.name == "test_torch_runtime.py" for item in session.items):
        return
    torch = sys.modules.get("torch")
    if torch is None or not hasattr(torch, "__version__"):
        return
    packages = {}
    for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "sentence-transformers", "scikit-learn", "numpy", "pyarrow", "pandas", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    reporter = session.config.pluginmanager.getplugin("terminalreporter")
    stats = reporter.stats if reporter else {}
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), exit_status=int(exitstatus),
                  collected=session.testscollected, passed=len(stats.get("passed", [])), failed=len(stats.get("failed", [])),
                  skipped=len(stats.get("skipped", [])), python=platform.python_version(), platform=platform.system(),
                  packages=packages, cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                  scope="Protocol tests and real PyTorch tiny/random-model runtime tests; no pretrained benchmark accuracy or A6000/NCCL qualification")
    root = Path(__file__).resolve().parents[1] / "verification"
    root.mkdir(exist_ok=True)
    (root / "pytorch_runtime_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
