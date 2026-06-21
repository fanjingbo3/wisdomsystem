from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class _LazyModel:
    """延迟初始化代理，首次使用时才真正创建模型实例"""
    def __init__(self, factory_method):
        self._factory_method = factory_method
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            self._instance = self._factory_method()
        return self._instance

    def materialize(self):
        """返回底层真实模型实例，用于需要isinstance检查的场景（如create_react_agent）"""
        return self._get_instance()

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)


class ChatModelFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 懒加载：首次调用时才真正初始化模型，避免启动时35s+的等待
chat_model = _LazyModel(ChatModelFactory().generator)

embed_model = _LazyModel(EmbeddingsFactory().generator)


def get_chat_model():
    """获取聊天模型实例（懒加载单例）"""
    return chat_model.materialize()


def get_embed_model():
    """获取嵌入模型实例（懒加载单例）"""
    return embed_model.materialize()
