from lib.kits.basic import *

from lib.data.datasets.hsmr_v1.mocap_dataset import MoCapDataset
from lib.data.datasets.hsmr_v1.wds_loader import load_tars_as_wds
import webdataset as wds

class DataModule(pl.LightningDataModule):
    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.cfg_eval = self.cfg.pop('eval', None)
        self.cfg_train = self.cfg.pop('train', None)
        self.cfg_mocap = self.cfg.pop('mocap', None)

    def setup(self, stage=None):
        if stage in ['test', None, '_debug_eval'] and self.cfg_eval is not None:
            self._setup_eval()
        if stage in ['fit', None, '_debug_train'] and self.cfg_train is not None:
            self._setup_train()
        if stage in ['fit', None, '_debug_mocap'] and self.cfg_mocap is not None:
            self._setup_mocap()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset = self.train_dataset,
            **self.cfg_train.dataloader,
        )

    # ========== Internal Modules to Setup Datasets ==========

    def _setup_train(self):
        names, datasets, weights = [], [], []
        ld_cfg = self.cfg_train.cfg  # cfg for initializing wds loading pipeline

        for ds_cfg in self.cfg_train.datasets:
            dataset = load_tars_as_wds(
                ld_cfg,
                ds_cfg.item.urls,
                ds_cfg.item.epoch_size
            )
            names.append(ds_cfg.name)
            datasets.append(dataset)
            weights.append(ds_cfg.weight)

        # Normalize the weights and mix the datasets.
        weights = to_numpy(weights)
        weights = weights / weights.sum()
        # 推荐直接用 RandomMix 返回的对象，不用 MixedWebDataset
        self.train_dataset = wds.RandomMix(datasets, weights).with_epoch(50_000).shuffle(1000, initial=1000)

    def _setup_mocap(self):
        self.mocap_dataset = MoCapDataset(**self.cfg_mocap.cfg)

    def _setup_eval(self, selected_ds_names:Optional[List[str]]=None):
        from lib.data.datasets.skel_hmr2_fashion.image_dataset import ImageDataset
        hack_cfg = {
            'IMAGE_SIZE': 256,
            'IMAGE_MEAN': [0.485, 0.456, 0.406],
            'IMAGE_STD': [0.229, 0.224, 0.225],
            'BBOX_SHAPE': [192, 256],
            'augm': self.cfg.image_augmentation,
            'SUPPRESS_KP_CONF_THRESH': 0.3,
            'FILTER_NUM_KP': 4,
            'FILTER_NUM_KP_THRESH': 0.0,
            'FILTER_REPROJ_THRESH': 31000,
            'SUPPRESS_BETAS_THRESH': 3.0,
            'SUPPRESS_BAD_POSES': False,
            'POSES_BETAS_SIMULTANEOUS': True,
            'FILTER_NO_POSES': False,
            'BETAS_REG': True,
        }
        self.eval_datasets = {}
        for dataset_cfg in self.cfg_eval.datasets:
            if selected_ds_names is not None and dataset_cfg.name not in selected_ds_names:
                continue
            dataset = ImageDataset(
                cfg=hack_cfg,
                dataset_file=dataset_cfg.item.dataset_file,
                img_dir=dataset_cfg.item.img_root,
                train=False,
            )
            dataset._kp_list_ = dataset_cfg.item.kp_list
            self.eval_datasets[dataset_cfg.name] = dataset
