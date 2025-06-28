# skelify-spin

定义SKELifySPIN回调过程 继承自/pipeline/skelify-refiner@skelify（Hydra支持的继承+定义模式）

i80：间隔幅度为80（大量减少伪标签生成数量，减少伪标签噪声干扰）
i10kb1：间隔幅度为10，新增max_batches_per_round参数，每次只更新最新伪标签（节约时间）
i230kb1：间隔幅度230+不做回溯（大量节省算力）