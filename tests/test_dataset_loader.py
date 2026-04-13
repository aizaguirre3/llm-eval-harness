from pathlib import Path

import pytest

from src.loaders.dataset import DatasetLoader, QAPair

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_data(tmp_path):
    data = [
        {
            "id": "t1",
            "question": "What is 2+2?",
            "expected_answer": "4",
            "context": "basic math",
            "metadata": {"category": "math", "difficulty": "easy"},
        },
        {
            "id": "t2",
            "question": "What is the capital of France?",
            "expected_answer": "Paris",
            "context": "geography",
            "metadata": {"category": "geography", "difficulty": "easy"},
        },
    ]
    import json

    (tmp_path / "test.json").write_text(json.dumps(data))
    return tmp_path


def test_load_returns_qa_pairs(sample_data):
    loader = DatasetLoader(data_dir=sample_data)
    pairs = loader.load("test.json")
    assert len(pairs) == 2
    assert all(isinstance(p, QAPair) for p in pairs)
    assert pairs[0].id == "t1"
    assert pairs[1].question == "What is the capital of France?"


def test_load_by_category(sample_data):
    loader = DatasetLoader(data_dir=sample_data)
    math_pairs = loader.load_by_category("test.json", "math")
    assert len(math_pairs) == 1
    assert math_pairs[0].id == "t1"


def test_load_missing_file(sample_data):
    loader = DatasetLoader(data_dir=sample_data)
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent.json")


def test_list_datasets(sample_data):
    loader = DatasetLoader(data_dir=sample_data)
    datasets = loader.list_datasets()
    assert "test.json" in datasets
