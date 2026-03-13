from ifcb_classify.data.datasets import build_transform, create_training_datasets


def test_build_transform_squarepad():
    t = build_transform("dataset_squarepad", 224, 224)
    assert t is not None


def test_create_training_datasets(tiny_imagefolder):
    data_dir, classes = tiny_imagefolder
    data = create_training_datasets(str(data_dir), "dataset_squarepad", 224, 224, val_split=0.2)
    assert "train" in data
    assert "val" in data
    assert data["num_classes"] == 3
    assert data["class_names"] == classes
    assert len(data["train"]) + len(data["val"]) == 15
