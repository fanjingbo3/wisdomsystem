from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ContextualEnhancer:
    def __init__(self):
        self._model = None
        self._chain = None

    def _get_model(self):
        if self._model is None:
            from model.factory import get_chat_model
            self._model = get_chat_model()
        return self._model

    def _get_chain(self):
        if self._chain is None:
            prompt_template = PromptTemplate.from_template("""
请为以下文档片段生成上下文描述，用于提升检索效果。

文档标题：{document_title}
文档片段：
{chunk_content}

请输出：【章节：xxx】【主题：xxx】【摘要：xxx】
""")
            self._chain = prompt_template | self._get_model() | StrOutputParser()
        return self._chain

    def enhance(self, document_title: str, chunk_content: str) -> str:
        try:
            context = self._get_chain().invoke({
                "document_title": document_title,
                "chunk_content": chunk_content
            })
            return f"{context}\n{chunk_content}"
        except Exception as e:
            return chunk_content