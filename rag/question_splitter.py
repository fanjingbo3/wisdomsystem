import re
from langchain_core.documents import Document


class QuestionBasedSplitter:
    def split_text(self, text: str) -> list[str]:
        question_pattern = r'(\d+)\.\s+\*\*([^*]+)\*\*'

        matches = list(re.finditer(question_pattern, text))

        if not matches:
            return self._split_by_paragraph(text)

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            chunk_text = text[start:end].strip()

            if len(chunk_text) > 500:
                sub_chunks = self._split_by_paragraph(chunk_text, max_len=500)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)

        chunks = self._add_overlap(chunks, overlap_ratio=0.2)

        return chunks

    def _split_by_paragraph(self, text: str, max_len: int = 500) -> list[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_len:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(para) > max_len:
                    sentences = self._split_by_sentence(para, max_len)
                    chunks.extend(sentences)
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_sentence(self, text: str, max_len: int = 500) -> list[str]:
        sentence_enders = ['。', '？', '！', '.', '?', '!']
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in sentence_enders:
                if len(current) >= max_len:
                    sentences.append(current.strip())
                    current = ""

        if current:
            sentences.append(current.strip())

        return sentences

    def _add_overlap(self, chunks: list[str], overlap_ratio: float = 0.2) -> list[str]:
        result = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_len = int(len(prev_chunk) * overlap_ratio)
                overlap_text = prev_chunk[-overlap_len:]
                chunk = overlap_text + "\n" + chunk
            result.append(chunk)
        return result

    def split_documents(self, documents: list[Document]) -> list[Document]:
        results = []
        chunk_id = 1

        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk_text in chunks:
                results.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "doc_id": f"chunk_{chunk_id:04d}",
                        "source": doc.metadata.get("source", ""),
                    }
                ))
                chunk_id += 1

        return results