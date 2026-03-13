import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from ifcb_classify.checkpoint import load_checkpoint
from ifcb_classify.config import InferConfig
from ifcb_classify.data.datasets import build_transform
from ifcb_classify.data.ifcb_bin import iter_bin_images, iter_directory_bins, get_bin_lid
from ifcb_classify.device import get_device
from ifcb_classify.hdf5_output import write_class_scores
from ifcb_classify.models.factory import get_model
from ifcb_classify.seed import set_seed

logger = logging.getLogger(__name__)


def infer_main(config: InferConfig) -> None:
    checkpoint = load_checkpoint(config.model_checkpoint, model_name=config.model_name, classes_path=config.classes_path)
    train_config = checkpoint["config"]
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    device = get_device(config.device)
    logger.info("Using device: %s", device)

    set_seed(train_config.get("seed", 42))
    model = get_model(train_config["model"], num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    transform = build_transform(
        train_config["transform"],
        train_config["image_width"],
        train_config["image_height"],
        train_config.get("mean"),
        train_config.get("std"),
    )

    thresholds = _load_thresholds(config, class_names)
    classifier_name = config.classifier_name or _derive_classifier_name(config, train_config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(config.input_path)
    if input_path.is_file():
        _classify_bin_file(input_path, model, transform, device, config.batch_size, class_names, thresholds, classifier_name, output_dir, config.overwrite)
    elif input_path.is_dir():
        _classify_directory(input_path, model, transform, device, config.batch_size, class_names, thresholds, classifier_name, output_dir, config.overwrite)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def _output_path_for_lid(output_dir: Path, lid: str) -> Path:
    return output_dir / f"{lid}_class.h5"


def _classify_bin_file(
    bin_path, model, transform, device, batch_size, class_names, thresholds, classifier_name, output_dir, overwrite
):
    lid = get_bin_lid(bin_path)
    out_path = _output_path_for_lid(output_dir, lid)

    if out_path.exists() and not overwrite:
        logger.info("Skipping (already exists): %s", out_path.name)
        return

    logger.info("Classifying bin: %s", lid)

    target_numbers = []
    images = []
    for target_num, img in iter_bin_images(bin_path):
        target_numbers.append(target_num)
        images.append(transform(img))

    if not images:
        logger.warning("No images in bin: %s", lid)
        return

    scores = _batch_predict(model, images, device, batch_size)
    _write_output(output_dir, lid, scores, class_names, target_numbers, classifier_name, thresholds)


def _classify_directory(
    dir_path, model, transform, device, batch_size, class_names, thresholds, classifier_name, output_dir, overwrite
):
    for lid, fbin in iter_directory_bins(dir_path):
        out_path = _output_path_for_lid(output_dir, lid)

        if out_path.exists() and not overwrite:
            logger.info("Skipping (already exists): %s", out_path.name)
            continue

        logger.info("Classifying bin: %s", lid)

        target_numbers = []
        images = []
        with fbin:
            for target_num in fbin.images.index:
                arr = fbin.images[target_num]
                from PIL import Image

                img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                target_numbers.append(target_num)
                images.append(transform(img))

        if not images:
            logger.warning("No images in bin: %s", lid)
            continue

        scores = _batch_predict(model, images, device, batch_size)
        _write_output(output_dir, lid, scores, class_names, target_numbers, classifier_name, thresholds)


def _batch_predict(model, images, device, batch_size):
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i : i + batch_size]).to(device)
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_scores.append(probs.cpu().numpy())
    return np.concatenate(all_scores, axis=0)


def _write_output(output_dir, lid, scores, class_names, target_numbers, classifier_name, thresholds):
    roi_numbers = np.array(target_numbers, dtype=np.int32)
    output_path = _output_path_for_lid(output_dir, lid)
    write_class_scores(output_path, scores, class_names, roi_numbers, classifier_name, thresholds)
    logger.info("Wrote: %s (%d ROIs)", output_path.name, len(target_numbers))


def _load_thresholds(config: InferConfig, class_names: list[str]) -> np.ndarray:
    path = config.thresholds_path

    # Auto-detect thresholds.json from model directory
    if not path:
        model_dir = Path(config.model_checkpoint).parent
        candidate = model_dir / "thresholds.json"
        if candidate.exists():
            logger.info("Auto-detected thresholds: %s", candidate)
            path = str(candidate)

    if path:
        if path.endswith(".json"):
            from ifcb_classify.thresholds import load_thresholds_json
            return load_thresholds_json(path, class_names)
        with open(path) as f:
            data = yaml.safe_load(f)
        return np.array([data.get(c, np.nan) for c in class_names], dtype=np.float64)

    return np.full(len(class_names), config.threshold_default, dtype=np.float64)


def _derive_classifier_name(config: InferConfig, train_config: dict) -> str:
    # For legacy checkpoints, use the model directory name
    model_dir = Path(config.model_checkpoint).parent
    dir_name = model_dir.name
    if dir_name and dir_name != ".":
        return dir_name
    return f"{train_config['model']}_{train_config.get('dataset_version', '')}"
