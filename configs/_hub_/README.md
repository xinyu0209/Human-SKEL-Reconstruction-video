# _hub

Configs here shouldn't be used as what Hydra calls a "config group". Configs here serve as a hub for other configs to reference. Most things here usually won't be used in a certain experiment.

For example, the details of each datasets can be defined here, and than I only need to reference the `Human3.6M` dataset through `${_hub_.datasets.h36m}` in other configs.

Each `.yaml` file here will be loaded separately in `base.yaml`. Check the `base.yaml` to understand how it works.

这里的配置不应该被用作Hydra所说的“配置组”。这里的配置作为一个集线器，供其他配置参考。这里的大多数东西通常不会在某个实验中使用。
例如，每个数据集的细节可以在这里定义，然后我只需要在其他配置中通过${_hub_.datasets.h36m}引用“human36M”数据集中的H36m
