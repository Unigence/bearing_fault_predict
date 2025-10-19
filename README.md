#### 文件夹内容

datasets - 划分后的数据集，包含训练集、验证集和测试集

models - 模型架构文件

preprocess - 预处理脚本，包含滤波和特征提取

raw_datasets - 官方提供的数据集源文件

runs - 训练结果和权重文件

tests - 测试结果

visualize - 可视化结果，保存数据集和预处理结果

---

#### 根目录脚本内容

dataset_loader.py - 从datasets中加载数据集

rawdata_processor.py - 预处理原始数据集到可以被加载的数据集，按512长、512间隔划分

rawdata_processor_direct.py - 预处理原始数据集到可以被加载的数据集，不进行窗口划分

test_launcher.py - 进行测试总入口

train_launcher.py - 进行训练总入口


---

#### 配置文件内容

data.yaml - 故障种类和数据集位置

configs.yaml - 模型和超参数配置

