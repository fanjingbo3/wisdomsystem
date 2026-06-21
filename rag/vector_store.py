from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os


class VectorStoreService:
    def __init__(self):
        self._vector_store = None
        self._embedding_model = None
        self._spliter = None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from model.factory import get_embed_model
            self._embedding_model = get_embed_model()
        return self._embedding_model

    def _get_spliter(self):
        if self._spliter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._spliter = RecursiveCharacterTextSplitter(
                chunk_size=chroma_conf["chunk_size"],
                chunk_overlap=chroma_conf["chunk_overlap"],
                separators=chroma_conf["separators"],
                length_function=len,
            )
        return self._spliter

    def _get_vector_store(self):
        if self._vector_store is None:
            from langchain_chroma import Chroma
            self._vector_store = Chroma(
                collection_name=chroma_conf["collection_name"],
                embedding_function=self._get_embedding_model(),
                persist_directory=chroma_conf["persist_directory"],
            )
        return self._vector_store

    def get_retriever(self):
        return self._get_vector_store().as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self._get_spliter().split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # ContextualEnhancer 已关闭以节省 embedding token
                # from rag.contextual_enhancer import ContextualEnhancer
                # enhancer = ContextualEnhancer()
                # document_title = split_document[0].metadata.get("source", "").split("/")[-1] if split_document else ""
                #
                # for doc in split_document:
                #     doc.page_content = enhancer.enhance(document_title, doc.page_content)

                existing_count = self._get_vector_store()._collection.count()
                for idx, doc in enumerate(split_document):
                    doc.metadata["doc_id"] = f"chunk_{existing_count + idx + 1:04d}"

                self._get_vector_store().add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

    def get_all_documents_with_ids(self) -> list:
        """获取所有文档及其 doc_id，用于测试用例标注"""
        collection = self._get_vector_store()._collection
        results = collection.get(include=["metadatas", "documents"])
        docs_with_ids = []
        for i in range(len(results["documents"])):
            docs_with_ids.append({
                "doc_id": results["metadatas"][i].get("doc_id", ""),
                "content": results["documents"][i][:200] + "..." if len(results["documents"][i]) > 200 else results["documents"][i],
                "full_content": results["documents"][i],
            })
        return sorted(docs_with_ids, key=lambda x: x["doc_id"])

    def _scan_md5_from_data(self, data_dir: str) -> list:
        """扫描 data 目录，生成 MD5 列表"""
        md5_list = []
        if not os.path.exists(data_dir):
            return md5_list

        allowed_types = tuple(chroma_conf.get("allow_knowledge_file_type", ["txt", "pdf"]))
        for path in listdir_with_allowed_type(data_dir, allowed_types):
            md5_hex = get_file_md5_hex(path)
            md5_list.append((md5_hex, path))

        return md5_list

    def delete_document(self, doc_id: str):
        """删除指定 doc_id 的文档，同时更新 MD5 记录"""
        # 重新获取 vector_store 确保连接最新
        self._vector_store = None
        vector_store = self._get_vector_store()
        collection = vector_store._collection
        # 从向量库删除，使用 where 条件
        collection.delete(where={"doc_id": doc_id})
        logger.info(f"[向量库] 已删除文档: {doc_id}")

        # 重新生成 MD5 文件（基于当前 data 目录中的文件）
        md5_file = get_abs_path(chroma_conf["md5_hex_store"])
        data_dir = get_abs_path(chroma_conf.get("data_path", "data"))

        new_md5_list = []
        for md5_hex, path in self._scan_md5_from_data(data_dir):
            if path and os.path.exists(path):
                new_md5_list.append(md5_hex)

        with open(md5_file, 'w') as f:
            f.write("\n".join(new_md5_list))

        logger.info(f"[向量库] 已更新MD5记录，共 {len(new_md5_list)} 个文件")

    def clear_collection(self):
        """清空向量库，用于重建"""
        self._get_vector_store()._collection.delete(ids=self._get_vector_store()._collection.get()["ids"])
        md5_file = get_abs_path(chroma_conf["md5_hex_store"])
        if os.path.exists(md5_file):
            os.remove(md5_file)
        logger.info("[向量库] 已清空向量库和MD5记录")


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    print("\n=== 所有文档及其 doc_id ===")
    docs = vs.get_all_documents_with_ids()
    for doc in docs:
        print(f"\n{doc['doc_id']}: {doc['content']}")
