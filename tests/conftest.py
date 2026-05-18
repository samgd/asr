import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
