"""The example notebooks in docs/examples: committed executed (the docs render their stored
outputs), and still running start to finish against the current code."""

import copy
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")
pytest.importorskip("matplotlib")

EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


@pytest.fixture(scope="session")
def kernel_dir(tmp_path_factory):
    """A kernelspec for the interpreter running the tests. Looking ``python3`` up by name
    would find whatever kernel the developer registered first, which may not have samplesar."""
    from ipykernel.kernelspec import install as install_kernelspec

    prefix = tmp_path_factory.mktemp("kernels")
    install_kernelspec(prefix=str(prefix), kernel_name="samplesar-tests")
    return prefix / "share" / "jupyter" / "kernels"


def test_the_notebooks_are_there_and_in_the_docs():
    assert [p.stem for p in NOTEBOOKS] == ["first_nest", "islands", "strategies"]
    nav = (EXAMPLES.parent.parent / "mkdocs.yml").read_text()
    for path in NOTEBOOKS:
        assert f"examples/{path.name}" in nav


# Seconds apiece (each starts a kernel), so "make test-fast" skips them.
@pytest.mark.slow
@pytest.mark.notebook
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_example_notebook_is_committed_executed_and_runs(path, kernel_dir):
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    committed = nbformat.read(path, as_version=4)
    code_cells = [c for c in committed.cells if c.cell_type == "code"]
    assert code_cells
    # What the docs render: every cell was run, none ended in a traceback or wrote warnings.
    assert all(c.execution_count is not None for c in code_cells), "notebook has unrun cells"
    assert not [o for c in code_cells for o in c.outputs if o.output_type == "error"]
    assert not [o for c in code_cells for o in c.outputs if o.get("name") == "stderr"]

    km = KernelManager(
        kernel_name="samplesar-tests",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]),
    )
    client = NotebookClient(
        copy.deepcopy(committed),
        km=km,
        timeout=600,
        resources={"metadata": {"path": str(EXAMPLES)}},
    )
    # NotebookClient leaves a kernel manager it was handed running when the run ends; have
    # it shut the kernel down and close its channels as soon as the notebook finishes or fails.
    try:
        fresh = client.execute(cleanup_kc=True)  # raises CellExecutionError on a failing cell
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    # The stored outputs are what the docs show: they must be what the code gives now.
    for before, after in zip(code_cells, _code_cells(fresh), strict=True):
        assert _text(before) == _text(after), (
            f"stale output in {path.name}; re-run and commit it:\n{before.source}"
        )


def _code_cells(notebook):
    return [c for c in notebook.cells if c.cell_type == "code"]


def _text(cell):
    """A cell's printed and returned text (not its images, whose bytes vary by platform)."""
    parts = []
    for output in cell.outputs:
        if output.output_type == "stream":
            parts.append(output.text)
        elif "text/plain" in output.get("data", {}) and "image/png" not in output.data:
            parts.append(output.data["text/plain"])
    return "".join(parts)
