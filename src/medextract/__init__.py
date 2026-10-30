"""MedExtract: biomedical entity extraction and retrieval.

The two calls at the bottom of this module are environment guards. Both defend
against failures that happen at import time, before any of this project's code
runs, and both are set here because this is the one module every entry point --
CLI, API, tests -- goes through, so they cannot drift out of sync with the
places that need them.
"""

import contextlib
import os

__version__ = "1.0.0"


def _preload_arrow() -> None:
    """Load Arrow's native runtime before torch loads its own.

    On Windows, importing pyarrow *after* torch and transformers have
    initialised takes down the process with an access violation -- not an
    exception, a segfault, exit code 139. It reproduces reliably: instantiate a
    transformers model, then let anything pull in `datasets` or `pandas`, and
    the interpreter dies inside `pyarrow/__init__.py`. Both libraries ship
    their own copies of overlapping native dependencies, and only the load
    order decides whether they coexist.

    The exposure is real rather than theoretical: `pipeline.py` imports torch
    to report the environment before `train()` gets as far as `import mlflow`,
    and mlflow pulls in pandas, which pulls in pyarrow.

    Inside the serving container this is a no-op: it carries neither pyarrow
    nor torch.
    """
    # Absent pyarrow is the expected case in the serving image, so suppress
    # rather than branch on it.
    with contextlib.suppress(Exception):
        import pyarrow  # noqa: F401


def _disable_tensorflow_integration() -> None:
    """Stop transformers from probing for TensorFlow.

    `transformers.Trainer` imports its reporting integrations, which reach for
    `TFPreTrainedModel`, which imports the TF activation helpers, which raise
    outright when the TensorFlow present is paired with Keras 3. Nothing in
    this project touches TensorFlow -- the models are PyTorch and the runtime
    is ONNX -- so a TF install that happens to be on the machine should have no
    say in whether training starts.

    The alternative is `pip install tf-keras`, which pulls in a compatibility
    shim purely to satisfy an import path that is never taken. Declining the
    path is both smaller and makes behaviour identical on machines that have
    TensorFlow and machines that do not.

    setdefault rather than assignment, so anyone who does want the TF path can
    still ask for it from the environment.
    """
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


_preload_arrow()
_disable_tensorflow_integration()
