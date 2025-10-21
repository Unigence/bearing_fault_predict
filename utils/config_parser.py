"""
配置文件解析器
统一解析model_config.yaml, train_config.yaml, augmentation_config.yaml
"""
import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigParser:
    """配置解析器基类"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default=None):
        """
        获取配置项(支持嵌套键,如'model.config')
        
        Args:
            key: 配置键,支持'.'分隔的嵌套键
            default: 默认值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置项(支持嵌套键)
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, save_path: str = None):
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径,默认为原路径
        """
        save_path = save_path or self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """返回完整配置字典"""
        return self.config


class ModelConfigParser(ConfigParser):
    """模型配置解析器"""
    
    def __init__(self, config_path: str = 'configs/model_config.yaml'):
        super().__init__(config_path)
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取模型构建参数"""
        return {
            'config': self.get('model.config', 'medium'),
            'seq_len': self.get('model.seq_len', 512),
            'num_classes': self.get('model.num_classes', 6),
            'fusion_type': self.get('model.fusion_type', 'hierarchical'),
            'head_type': self.get('model.head_type', 'dual')
        }
    
    def get_backbone_params(self) -> Dict[str, Any]:
        """获取backbone参数"""
        return {
            'temporal': self.get('backbone.temporal', {}),
            'frequency': self.get('backbone.frequency', {}),
            'timefreq': self.get('backbone.timefreq', {})
        }
    
    def get_checkpoint_params(self) -> Dict[str, Any]:
        """获取checkpoint参数"""
        return self.get('checkpoint', {})
    
    def get_pretrain_params(self) -> Dict[str, Any]:
        """获取预训练加载参数"""
        return self.get('pretrain', {})


class TrainConfigParser(ConfigParser):
    """训练配置解析器"""
    
    def __init__(self, config_path: str = 'configs/train_config.yaml'):
        super().__init__(config_path)
    
    def get_training_mode(self) -> str:
        """获取训练模式: 'pretrain_finetune' 或 'direct_train'"""
        return self.get('training_mode.pipeline', 'pretrain_finetune')
    
    def use_pretrain(self) -> bool:
        """是否使用预训练阶段"""
        return self.get_training_mode() == 'pretrain_finetune'
    
    def use_mixup(self) -> bool:
        """是否使用mixup"""
        return self.get('training_mode.use_mixup', True)
    
    def get_pretrain_params(self) -> Dict[str, Any]:
        """获取预训练参数"""
        return self.get('pretrain', {})
    
    def get_finetune_params(self) -> Dict[str, Any]:
        """获取微调参数"""
        return self.get('finetune', {})
    
    def get_optimizer_params(self, stage: str = 'finetune') -> Dict[str, Any]:
        """
        获取优化器参数
        
        Args:
            stage: 'pretrain' 或 'finetune'
        """
        return self.get(f'{stage}.optimizer', {})
    
    def get_scheduler_params(self, stage: str = 'finetune') -> Dict[str, Any]:
        """
        获取学习率调度器参数
        
        Args:
            stage: 'pretrain' 或 'finetune'
        """
        return self.get(f'{stage}.scheduler', {})
    
    def get_loss_params(self, stage: str = 'finetune') -> Dict[str, Any]:
        """
        获取损失函数参数
        
        Args:
            stage: 'pretrain' 或 'finetune'
        """
        return self.get(f'{stage}.loss', {})
    
    def get_data_params(self) -> Dict[str, Any]:
        """获取数据加载参数"""
        return self.get('data', {})
    
    def get_experiment_params(self) -> Dict[str, Any]:
        """获取实验管理参数"""
        return self.get('experiment', {})
    
    def get_device(self) -> str:
        """获取训练设备"""
        return self.get('device.type', 'cuda')
    
    def use_amp(self) -> bool:
        """是否使用混合精度训练"""
        return self.get('device.use_amp', False)
    
    def get_seed(self) -> int:
        """获取随机种子"""
        return self.get('seed', 42)


class AugmentationConfigParser(ConfigParser):
    """数据增强配置解析器"""
    
    def __init__(self, config_path: str = 'configs/augmentation_config.yaml'):
        super().__init__(config_path)

    def enable_augmentation(self) -> bool:
        """
        Returns:
            True=启用数据增强, False=禁用数据增强
        """
        return self.get('progressive.enable_augmentation', True)

    def use_progressive(self) -> bool:
        """是否使用渐进式增强"""
        # 如果数据增强被禁用,则返回False
        if not self.enable_augmentation():
            return False

        # 如果启用了恒定增强,则不使用渐进式
        if self.use_constant():
            return False

        return self.get('progressive.enable', True)
    
    def use_constant(self) -> bool:
        """是否使用恒定增强"""
        if not self.enable_augmentation():
            return False
        return self.get('progressive.use_constant', False)
    
    def get_default_intensity(self) -> str:
        """获取默认增强强度"""
        return self.get('progressive.default_intensity', 'medium')
    
    def get_contrastive_aug_params(self) -> Dict[str, Any]:
        """获取对比学习增强参数"""
        return self.get('contrastive', {})
    
    def get_supervised_aug_params(self, intensity: str = 'medium') -> Dict[str, Any]:
        """
        获取有监督增强参数
        
        Args:
            intensity: 'weak', 'medium', 'strong'
        """
        return self.get(f'supervised.{intensity}', {})
    
    def get_mixup_params(self) -> Dict[str, Any]:
        """获取mixup参数"""
        return self.get('mixup', {})


def load_all_configs(config_dir: str = 'configs') -> Dict[str, ConfigParser]:
    """
    加载所有配置文件
    
    Args:
        config_dir: 配置文件目录
    
    Returns:
        {
            'model': ModelConfigParser,
            'train': TrainConfigParser,
            'augmentation': AugmentationConfigParser
        }
    """
    config_dir = Path(config_dir)
    
    configs = {
        'model': ModelConfigParser(str(config_dir / 'model_config.yaml')),
        'train': TrainConfigParser(str(config_dir / 'train_config.yaml')),
        'augmentation': AugmentationConfigParser(str(config_dir / 'augmentation_config.yaml'))
    }
    
    return configs


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典
    
    Args:
        *configs: 配置字典列表
    
    Returns:
        merged_config: 合并后的配置字典
    """
    merged = {}
    
    for config in configs:
        _deep_update(merged, config)
    
    return merged


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
    """深度更新字典"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


if __name__ == '__main__':
    """测试配置解析器"""
    print("=" * 70)
    print("配置解析器测试")
    print("=" * 70)
    
    # 测试模型配置解析
    print("\n1. 模型配置解析")
    try:
        model_config = ModelConfigParser()
        print(f"   模型参数: {model_config.get_model_params()}")
        print(f"   Checkpoint参数: {model_config.get_checkpoint_params()}")
    except FileNotFoundError as e:
        print(f"   ⚠️  {e}")
    
    # 测试训练配置解析
    print("\n2. 训练配置解析")
    try:
        train_config = TrainConfigParser()
        print(f"   训练模式: {train_config.get_training_mode()}")
        print(f"   使用预训练: {train_config.use_pretrain()}")
        print(f"   设备: {train_config.get_device()}")
        print(f"   随机种子: {train_config.get_seed()}")
    except FileNotFoundError as e:
        print(f"   ⚠️  {e}")
    
    # 测试增强配置解析
    print("\n3. 增强配置解析")
    try:
        aug_config = AugmentationConfigParser()
        print(f"   渐进式增强: {aug_config.use_progressive()}")
        print(f"   恒定增强: {aug_config.use_constant()}")
        print(f"   默认强度: {aug_config.get_default_intensity()}")
    except FileNotFoundError as e:
        print(f"   ⚠️  {e}")
    
    # 测试加载所有配置
    print("\n4. 加载所有配置")
    try:
        all_configs = load_all_configs()
        print(f"   ✓ 成功加载 {len(all_configs)} 个配置文件")
    except Exception as e:
        print(f"   ⚠️  {e}")
    
    print("\n" + "=" * 70)
    print("✓ 配置解析器测试完成")
    print("=" * 70)
