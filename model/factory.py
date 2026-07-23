from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from utils.config_handler import rag_conf


class TokenTrackingChatTongyi(ChatTongyi):
    """增强版 ChatTongyi，正确传递 token_usage 到 LangSmith"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = {
            'ls_provider': 'dashscope',
            'ls_model_name': self.model_name,
        }
    
    def _generate(
        self,
        messages: List[BaseChatModel],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        llm_output = {}
        if result.generations:
            for generation in result.generations:
                if isinstance(generation, ChatGeneration):
                    tu = {}
                    if generation.generation_info and 'token_usage' in generation.generation_info:
                        tu = generation.generation_info['token_usage']
                    elif generation.message.response_metadata and 'token_usage' in generation.message.response_metadata:
                        tu = generation.message.response_metadata['token_usage']
                    
                    if tu:
                        usage_metadata = {
                            'input_tokens': tu.get('input_tokens', tu.get('prompt_tokens', 0)),
                            'output_tokens': tu.get('output_tokens', tu.get('completion_tokens', 0)),
                            'total_tokens': tu.get('total_tokens', 0),
                        }
                        generation.message.usage_metadata = usage_metadata
                        llm_output = {
                            'token_usage': {
                                'prompt_tokens': usage_metadata['input_tokens'],
                                'completion_tokens': usage_metadata['output_tokens'],
                                'total_tokens': usage_metadata['total_tokens'],
                            },
                            'model_name': self.model_name,
                        }
        
        result.llm_output = llm_output
        return result
    
    def _stream(
        self,
        messages: List[BaseChatModel],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ):
        tu = {}
        for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            if chunk.message.response_metadata and 'token_usage' in chunk.message.response_metadata:
                tu = chunk.message.response_metadata['token_usage']
            yield chunk
        
        if tu:
            usage_metadata = {
                'input_tokens': tu.get('input_tokens', tu.get('prompt_tokens', 0)),
                'output_tokens': tu.get('output_tokens', tu.get('completion_tokens', 0)),
                'total_tokens': tu.get('total_tokens', 0),
            }
            chunk.message.usage_metadata = usage_metadata


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
        return TokenTrackingChatTongyi(model=rag_conf["chat_model_name"])


class LightChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        return TokenTrackingChatTongyi(model=rag_conf.get("light_chat_model_name", "qwen-turbo"))


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 懒加载：首次调用时才真正初始化模型，避免启动时35s+的等待
chat_model = _LazyModel(ChatModelFactory().generator)

light_chat_model = _LazyModel(LightChatModelFactory().generator)

embed_model = _LazyModel(EmbeddingsFactory().generator)


def get_chat_model():
    """获取聊天模型实例（懒加载单例）"""
    return chat_model.materialize()


def get_light_chat_model():
    """获取轻量聊天模型实例（qwen-turbo，用于路由分类和简单问题直接回答）"""
    return light_chat_model.materialize()


def get_embed_model():
    """获取嵌入模型实例（懒加载单例）"""
    return embed_model.materialize()
