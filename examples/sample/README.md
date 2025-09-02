# Sample Examples - 文件功能说明

## � Python文件功能
### `dataset_collect.py` - 数据集收集脚本
- 扩展的数据采集功能
- 支持更复杂的数据收集策略
- 集成多种数据收集器

### `dataset_info.py` - 数据集信息查看脚本
- 扩展的数据采集功能
- 支持更复杂的数据收集策略
- 集成多种数据收集器

### `extract_episode_images.py` - 图像提取脚本
- 从zarr数据集中提取episode的相机图像
- 保存为PNG格式用于视频制作
- 支持选择特定episode和图像通道
- 
### `filter_episodes_by_length.py` - Episode长度过滤脚本
- 从zarr数据集中过滤掉长度小于指定值的episode
- 保存为新的数据集文件
- 支持指定要保留的数据键和长度阈值


### `make_video.py` - 视频制作工具
- 将PNG帧序列转换为MP4视频
- 可调节帧率
- 自动处理帧文件排序

