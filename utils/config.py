import yaml
from pathlib import Path


class Config:
    def __init__(self, model_config_path, train_config_path):
        self.model_config_path = Path(model_config_path)
        self.train_config_path = Path(train_config_path)
        self.model_config = self._load_model_config()
        self.train_config = self._load_train_config()

    def _load_model_config(self):
        """加载MODEL配置文件"""
        with open(self.model_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
        
    def _load_train_config(self):
        """加载TRAIN配置文件"""
        with open(self.train_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_backbone_config(self):
        """获取backbone相关配置"""
        return self.model_config.get('backbone', {})

    def get_detector_config(self):
        """获取detector相关配置"""
        return self.model_config.get('detector', {})

    def get_temporal_config(self):
        """获取temporal相关配置"""
        return self.model_config.get('temporal', {})

    def get_training_config(self):
        """获取training相关配置"""
        return self.train_config.get('training', {})