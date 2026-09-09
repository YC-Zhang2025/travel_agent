from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "travel_attractions"

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """创建并缓存本地中文 Embedding 模型。"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """连接已构建的本地 Chroma 索引。"""
    if not CHROMA_DIRECTORY.exists():
        raise FileNotFoundError(
            "没有找到 Chroma 索引，请先运行："
            "python scripts/build_index.py"
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIRECTORY),
    )


def retrieve_travel_knowledge(
    query: str,
    city: str | None = None,
    limit: int = 3,
) -> list[dict]:
    """根据自然语言查询检索旅游知识。"""
    vector_store = get_vector_store()

    search_arguments = {
        "query": QUERY_INSTRUCTION + query,
        "k": limit,
    }

    if city:
        search_arguments["filter"] = {"city": city}

    documents = vector_store.similarity_search(
        **search_arguments
    )

    return [
        {
            "content": document.page_content,
            "city": document.metadata.get("city"),
            "name": document.metadata.get("name"),
            "category": document.metadata.get("category"),
            "ticket": document.metadata.get("ticket"),
            "source": document.metadata.get("source"),
        }
        for document in documents
    ]