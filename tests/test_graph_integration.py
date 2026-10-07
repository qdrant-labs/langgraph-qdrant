from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from langgraph.store.qdrant import AsyncQdrantStore, QdrantStore
from tests.conftest import DIMS
from tests.embed_test_utils import CharacterEmbeddings


class State(TypedDict):
    message: str
    recalled: Annotated[list[str], operator.add]


class Context(TypedDict):
    user_id: str


def _build(store: Any) -> Any:
    def remember(state: State, runtime: Runtime[Context]) -> dict:
        ns = ("memories", runtime.context["user_id"])
        assert runtime.store is not None
        runtime.store.put(ns, str(uuid.uuid4()), {"text": state["message"]})
        return {}

    def recall(state: State, runtime: Runtime[Context]) -> dict:
        ns = ("memories", runtime.context["user_id"])
        assert runtime.store is not None
        hits = runtime.store.search(ns, query=state["message"], limit=1)
        return {"recalled": [h.value["text"] for h in hits]}

    async def aremember(state: State, runtime: Runtime[Context]) -> dict:
        ns = ("memories", runtime.context["user_id"])
        assert runtime.store is not None
        await runtime.store.aput(ns, str(uuid.uuid4()), {"text": state["message"]})
        return {}

    async def arecall(state: State, runtime: Runtime[Context]) -> dict:
        ns = ("memories", runtime.context["user_id"])
        assert runtime.store is not None
        hits = await runtime.store.asearch(ns, query=state["message"], limit=1)
        return {"recalled": [h.value["text"] for h in hits]}

    is_async = isinstance(store, AsyncQdrantStore)
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("recall", arecall if is_async else recall)
    builder.add_node("remember", aremember if is_async else remember)
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "remember")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=InMemorySaver(), store=store)


def test_memories_shared_across_threads(make_store: Any) -> None:
    store: QdrantStore = make_store(
        index={"dims": DIMS, "embed": CharacterEmbeddings(DIMS), "fields": ["text"]}
    )
    graph = _build(store)

    def run(thread: str, user: str, message: str) -> list[str]:
        out = graph.invoke(
            {"message": message, "recalled": []},
            {"configurable": {"thread_id": thread}},
            context={"user_id": user},
        )
        return out["recalled"]

    assert run("t1", "alice", "I love hiking in the alps") == []
    assert run("t2", "alice", "I love hiking") == ["I love hiking in the alps"]
    assert run("t3", "bob", "I love hiking") == []
    assert store.list_namespaces(prefix=("memories",)) == [
        ("memories", "alice"),
        ("memories", "bob"),
    ]


async def test_memories_shared_across_threads_async(make_astore: Any) -> None:
    store: AsyncQdrantStore = await make_astore(
        index={"dims": DIMS, "embed": CharacterEmbeddings(DIMS), "fields": ["text"]}
    )
    graph = _build(store)

    async def run(thread: str, user: str, message: str) -> list[str]:
        out = await graph.ainvoke(
            {"message": message, "recalled": []},
            {"configurable": {"thread_id": thread}},
            context={"user_id": user},
        )
        return out["recalled"]

    assert await run("t1", "alice", "my cat is called Miso") == []
    assert await run("t2", "alice", "cat name") == ["my cat is called Miso"]
    assert await store.alist_namespaces() == [("memories", "alice")]
