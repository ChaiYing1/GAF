from dataclasses import fields

from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_gaf import DatasetGAF, DatasetGAFCfg, DatasetGAFCfgWrapper, DatasetDL3DVCfgWrapper, \
    DatasetScannetppCfgWrapper
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "gaf": DatasetGAF,
    "dl3dv": DatasetGAF,
    "scannetpp": DatasetGAF,
}


DatasetCfgWrapper = DatasetGAFCfgWrapper | DatasetDL3DVCfgWrapper | DatasetScannetppCfgWrapper
DatasetCfg = DatasetGAFCfg


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
) -> list[Dataset]:
    datasets = []
    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        datasets.append(dataset)

    return datasets
