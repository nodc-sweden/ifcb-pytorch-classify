import shutil
from pathlib import Path


from ifcb_classify.data.datasets import create_training_datasets, filter_classes

FIXTURES = Path(__file__).parent / "fixtures" / "training_data"


def test_create_training_datasets_from_fixtures():
    data = create_training_datasets(
        data_dir=str(FIXTURES),
        transform_name="dataset_squarepad",
        width=32,
        height=32,
        val_split=0.2,
    )
    assert data["num_classes"] == 2
    assert set(data["class_names"]) == {"Mesodinium_major", "Strombidium-like"}
    assert len(data["train"]) + len(data["val"]) == 30


def test_training_dataset_outputs_correct_shape():
    data = create_training_datasets(
        data_dir=str(FIXTURES),
        transform_name="dataset_squarepad",
        width=32,
        height=32,
    )
    img, label = data["train"][0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)


def test_filter_classes_min_images(tmp_path):
    """Copy fixtures to tmp so symlinks work on any filesystem."""
    data_dir = tmp_path / "training_data"
    shutil.copytree(FIXTURES, data_dir)

    # Mesodinium_major has 14 images, Strombidium-like has 16
    filtered_dir, filtered = filter_classes(str(data_dir), min_images=15)
    assert "Strombidium-like" in filtered
    assert "Mesodinium_major" not in filtered


def test_filter_classes_manual_include(tmp_path):
    data_dir = tmp_path / "training_data"
    shutil.copytree(FIXTURES, data_dir)

    filtered_dir, filtered = filter_classes(
        str(data_dir), min_images=100, manual_include=["Mesodinium_major"],
    )
    assert "Mesodinium_major" in filtered


def test_all_non_normalised_transforms_work():
    non_normalised = [
        "dataset", "dataset_squarepad", "dataset_fullpad",
        "dataset_squarepad_augmented", "dataset_fullpad_augmented",
    ]
    for name in non_normalised:
        data = create_training_datasets(
            data_dir=str(FIXTURES),
            transform_name=name,
            width=32,
            height=32,
        )
        img, _ = data["train"][0]
        assert img.shape == (3, 32, 32), f"Transform {name} produced wrong shape"
