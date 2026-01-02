# ACE Playbook

Agentic Context Engineering (ACE) is a three-role loop—**Generator**, **Reflector**, and **Curator**—that evolves a structured playbook of bullets describing strategies, rules, pitfalls, and tools. This repository delivers a production-ready implementation inspired by *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models* (arXiv:2510.04618).

## Architecture Overview

```
┌──────────┐    traces    ┌────────────┐    delta    ┌──────────┐
│ Generator├──────────────► Reflector  ├────────────► Curator   │
└────┬─────┘                └────┬─────┘             └────┬─────┘
     │ retrieval bullets         │ lessons                 │ merge+refine
     ▼                           ▼                         ▼
 Structured Playbook       Delta JSON (bullets)     SQLite + embeddings
```

* **Generator** retrieves the top-K bullets, solves tasks, and records traces with token usage and success signals.
* **Reflector** converts traces into *delta updates*: new bullets plus edits (counter increments, short patches).
* **Curator** deterministically merges deltas into SQLite, performs semantic de-duplication, and enforces the grow-and-refine policy (lazy or proactive).

### Delta Updates

* All context evolution happens through itemized bullets and patches.
* Semantic de-duplication uses cosine similarity with configurable thresholds.
* Helpful/harmful counters capture observed utility and guide ranking.

### Grow-and-Refine Policies

* **Proactive**: refine/prune immediately after every merge.
* **Lazy**: only prune when the playbook exceeds a configured window.

## Quickstart

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
# optional: export ACE_BASE_URL=http://localhost:8000/v1
# optional: export ACE_EMBEDDING_BASE_URL=http://localhost:8000/v1

python -m cli.ace_offline train data/train.csv --epochs 3
python -m cli.ace_online rollout data/test.csv
python -m cli.ace_playbook retrieve "How do I price a bond?"
python -m cli.ace_offline export ace_playbook.sqlite --output-path playbook.json
```

The offline loop optimizes contexts on a training split, while the online loop adapts during evaluation episodes. CSV
inputs should include `question` and `answer` columns (optional `evaluator` and `tolerance` columns are supported).

## Storage & Retrieval

* Persistent SQLite database managed via SQLAlchemy.
* Bullet embeddings stored as binary blobs with OpenAI-compatible or local sentence-transformers backends.
* Retrieval uses hybrid scoring: embedding similarity, helpful/harmful counters, and freshness bonuses.

## Pipelines & Evaluation

* `pipeline_offline.py` performs multi-epoch adaptation over CSV QA datasets.
* `pipeline_online.py` executes per-episode updates for self-improving agents.
* `evaluation.py` computes accuracy and token usage summaries.

## Examples

* `examples/online_agent_loop.py` demonstrates an online agent updating the playbook while solving synthetic math tasks.
* `examples/offline_optimize_finance.ipynb` outlines how to run offline adaptation on domain CSVs.

## Extending

* Implement new task adapters by yielding `(query, answer)` pairs.
* Integrate tool-calling by extending `llm_client.py` and updating the generator prompt.
* Configure dedup thresholds, retrieval weights, and grow-and-refine mode via environment variables prefixed with `ACE_`.

## Testing

Run the automated test suite:

```bash
pytest
```

## License

MIT License.
