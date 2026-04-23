import json
import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import get_logger


class RAGPipeline:
    def __init__(
        self,
        knowledge_path: str,
        persist_dir: str = "data/chroma_db",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        default_k: int = 3,
    ) -> None:
        self.logger = get_logger("RAGPipeline")
        self.knowledge_path = Path(knowledge_path)
        self.persist_dir = Path(persist_dir)
        self.default_k = default_k

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectorstore = self._load_or_create_vectorstore()

        self.answer_llm = None
        if os.getenv("OPENAI_API_KEY"):
            self.answer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    def _load_or_create_vectorstore(self) -> Chroma:
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        documents = self._load_and_chunk_documents()
        ids = [f"{doc.metadata.get('id', 'chunk')}_{i}" for i, doc in enumerate(documents)]
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name="autostream_knowledge",
            ids=ids,
        )

    def _load_and_chunk_documents(self) -> List[Document]:
        if not self.knowledge_path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {self.knowledge_path}")

        with self.knowledge_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        source_docs: List[Document] = []
        for record in records:
            source_docs.append(
                Document(
                    page_content=record["content"],
                    metadata={
                        "id": record.get("id", ""),
                        "title": record.get("title", ""),
                        "category": record.get("category", ""),
                        "source": record.get("source", ""),
                    },
                )
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=40)
        chunked_docs = splitter.split_documents(source_docs)
        self.logger.info("Loaded %d knowledge chunks into vector index", len(chunked_docs))
        return chunked_docs

    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        final_k = k or self.default_k
        scored_docs = self.vectorstore.similarity_search_with_relevance_scores(query, k=final_k)
        filtered = [(doc, score) for doc, score in scored_docs if score >= 0.35]
        return filtered

    def answer_with_context(self, query: str, k: int = None) -> Tuple[str, List[str]]:
        retrieved = self.retrieve(query, k=k)
        if not retrieved:
            return "I don't have that information", []

        chunks = []
        for doc, _ in retrieved:
            if doc.page_content not in chunks:
                chunks.append(doc.page_content)
        context_block = "\n".join([f"- {chunk}" for chunk in chunks])

        if self.answer_llm is None:
            # Deterministic non-hallucinating fallback based strictly on retrieved chunks.
            answer = "Based on our documented information:\n" + "\n".join(
                [f"- {chunk}" for chunk in chunks]
            )
            return answer, chunks

        prompt = f"""
You are an AutoStream assistant.
You MUST answer only from the context below.
If the context does not contain enough information, output exactly: I don't have that information

Context:
{context_block}

User question:
{query}
""".strip()

        response = self.answer_llm.invoke(prompt).content.strip()
        if not response:
            response = "I don't have that information"

        return response, chunks
