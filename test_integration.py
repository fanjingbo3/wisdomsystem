"""
测试向量库、MD5、doc_id 三者关联性
"""
import sys
sys.path.insert(0, '.')

import os
import hashlib
from rag.vector_store import VectorStoreService

def test_integration():
    # 测试文本
    test_content = "这是测试充电功能的文档片段"

    # ========== 1. 写入测试 ==========
    print("=" * 60)
    print("步骤1: 写入测试文档")
    print("=" * 60)

    # 检查 MD5 前状态
    md5_file = "md5.txt"
    md5_before = None
    if os.path.exists(md5_file):
        with open(md5_file, 'r') as f:
            md5_before = f.read().strip()
    print(f"写入前MD5: {md5_before or '无'}")

    # 手动添加测试文档到向量库
    from langchain_core.documents import Document
    from rag.contextual_enhancer import ContextualEnhancer

    vs = VectorStoreService()
    enhancer = ContextualEnhancer()

    # 增强内容
    enhanced_content = enhancer.enhance("测试文档", test_content)
    print(f"增强后内容: {enhanced_content[:80]}...")

    # 创建文档
    test_doc = Document(
        page_content=enhanced_content,
        metadata={"source": "test_file.txt"}
    )

    # 获取当前 doc_id
    existing_count = vs._get_vector_store()._collection.count()
    test_doc.metadata["doc_id"] = f"chunk_{existing_count + 1:04d}"
    print(f"分配的doc_id: {test_doc.metadata['doc_id']}")

    # 添加到向量库
    vs._get_vector_store().add_documents([test_doc])

    # 计算并保存MD5
    content_md5 = hashlib.md5(test_content.encode()).hexdigest()
    with open(md5_file, 'a') as f:
        f.write("\n" + content_md5)

    print(f"计算的MD5: {content_md5}")

    # ========== 2. 验证 ==========
    print("\n" + "=" * 60)
    print("步骤2: 验证存储")
    print("=" * 60)

    # 检查向量库
    docs = vs.get_all_documents_with_ids()
    print(f"向量库文档总数: {len(docs)}")

    # 查找测试文档
    test_doc_found = None
    for doc in docs:
        if "chunk_" + str(existing_count + 1).zfill(4) in doc['doc_id']:
            test_doc_found = doc
            break

    if test_doc_found:
        print(f"找到测试文档 doc_id: {test_doc_found['doc_id']}")
        print(f"内容预览: {test_doc_found['content'][:80]}...")
    else:
        print("未找到测试文档")

    # 检查MD5文件
    with open(md5_file, 'r') as f:
        md5_after = f.read().strip()
    print(f"写入后MD5文件内容: {md5_after[-64:]}...")

    # ========== 3. 删除测试 ==========
    print("\n" + "=" * 60)
    print("步骤3: 删除测试文档")
    print("=" * 60)

    if test_doc_found:
        doc_id = test_doc_found['doc_id']
        vs.delete_document(doc_id)
        print(f"已从向量库删除: {doc_id}")

        # 移除MD5记录（简化处理，实际需要记录位置）
        # 这里简化：重新加载文档时会通过MD5检查

    # ========== 4. 验证删除 ==========
    print("\n" + "=" * 60)
    print("步骤4: 验证删除")
    print("=" * 60)

    docs_after = vs.get_all_documents_with_ids()
    print(f"删除后向量库文档数: {len(docs_after)}")

    test_doc_after = None
    for doc in docs_after:
        if "chunk_" + str(existing_count + 1).zfill(4) in doc['doc_id']:
            test_doc_after = doc
            break

    if test_doc_after:
        print("❌ 测试文档仍然存在")
    else:
        print("✅ 测试文档已删除")

    print("\n测试完成！")

if __name__ == '__main__':
    test_integration()
