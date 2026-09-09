import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_FILE = PROJECT_ROOT / "data" / "knowledge" / "attractions.json"
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "travel_attractions"


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents() -> tuple[list[Document], list[str]]:
    with KNOWLEDGE_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    documents = []
    document_ids = []

    for record in records:
        documents.append(
            Document(
                page_content=record["content"],
                metadata={
                    "city": record["city"],
                    "name": record["name"],
                    "category": record["category"],
                    "ticket": record["ticket"],
                    "source": record["source"],
                },
            )
        )
        document_ids.append(record["id"])

    return documents, document_ids


def main() -> None:
    documents, document_ids = load_documents()
    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIRECTORY),
    )

    vector_store.add_documents(
        documents=documents,
        ids=document_ids,
    )

    print(f"索引构建完成，共写入 {len(documents)} 篇文档")
    print(f"索引目录：{CHROMA_DIRECTORY}")

    query = "为这个句子生成表示以用于检索相关文章：成都适合美食爱好者的景点"

    results = vector_store.similarity_search(
        query=query,
        k=3,
        filter={"city": "成都"},
    )

    print("\n--- 检索测试 ---")
    for index, document in enumerate(results, start=1):
        print(
            f"{index}. {document.metadata['name']} | "
            f"类别={document.metadata['category']} | "
            f"来源={document.metadata['source']}"
        )
        print(f"   {document.page_content}")


if __name__ == "__main__":
    main()