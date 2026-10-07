# langgraph-store-qdrant

A [Qdrant](https://qdrant.tech) store for
[LangGraph](https://github.com/langchain-ai/langgraph). It gives your agents
long-term memory: data they save in one conversation and recall in later ones.

It plugs in anywhere LangGraph accepts a store, such as
`graph.compile(store=...)`, in place of `InMemoryStore` or `PostgresStore`.

## Installation

```bash
uv add langgraph-store-qdrant
```

## Usage

```python
from qdrant_client import QdrantClient
from langgraph.store.qdrant import QdrantStore

store = QdrantStore(QdrantClient(url="http://localhost:6333"))
store.setup()  # run once

store.put(("users", "alice"), "prefs", {"theme": "dark", "lang": "en"})
store.get(("users", "alice"), "prefs").value
store.search(("users",), filter={"lang": "en"})
store.list_namespaces(prefix=("users",))
store.delete(("users", "alice"), "prefs")
```

### Semantic search

```python
from langchain.embeddings import init_embeddings

with QdrantStore.from_url(
    "http://localhost:6333",
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text"],  # defaults to the whole value
    },
) as store:
    store.setup()
    store.put(("memories", "alice"), "m1", {"text": "Alice loves hiking"})
    store.put(("memories", "alice"), "m2", {"text": "Alice is vegetarian"})

    for hit in store.search(("memories", "alice"), query="food preferences"):
        print(hit.key, hit.score, hit.value)
```

`index` also accepts `distance` (`"cosine"`, `"dot"`, `"euclid"` or
`"manhattan"`) and Qdrant's `hnsw_config`, `quantization_config`, `on_disk` and
`search_params`. Pass `index=False` to `put` to store an item without
embedding it, or `index=["title"]` to embed other fields.

### Async

```python
from langgraph.store.qdrant import AsyncQdrantStore

async with AsyncQdrantStore.from_url("http://localhost:6333") as store:
    await store.setup()
    await store.aput(("users", "alice"), "prefs", {"theme": "dark"})
    item = await store.aget(("users", "alice"), "prefs")
```

### In a graph

```python
def remember(state: State, runtime: Runtime[Context]):
    namespace = ("memories", runtime.context["user_id"])
    memories = runtime.store.search(namespace, query=state["message"], limit=3)
    runtime.store.put(namespace, str(uuid.uuid4()), {"text": state["message"]})
    ...

graph = builder.compile(checkpointer=checkpointer, store=store)
```

### Expiry

```python
store = QdrantStore(
    client,
    ttl={
        "default_ttl": 60 * 24,  # minutes
        "refresh_on_read": True,
        "omit_expired": True,
        "sweep_interval_minutes": 10,
    },
)
store.put(("cache",), "k", {"v": 1}, ttl=5)
store.start_ttl_sweeper()  # or call store.sweep_ttl() yourself
```

## LICENSE

[Apache 2.0](./LICENSE)
