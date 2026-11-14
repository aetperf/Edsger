# Ranking the vertices of a network

This example ranks the pages of a small synthetic web graph with `Centrality`, after a look at its structure with `ConnectedComponents`. The results are Polars DataFrames, which are joined and sorted.

## The graph

Each new page links to 3 existing pages, chosen with a probability that grows with the number of links they already receive (preferential attachment), and a few random links are added, some of them backwards, so that the graph has cycles.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
n_pages, n_links = 2_000, 3

in_degree = np.ones(n_pages)
tails, heads = [], []
for page in range(n_links, n_pages):
    weights = in_degree[:page] / in_degree[:page].sum()
    targets = rng.choice(page, size=n_links, replace=False, p=weights)
    tails += [page] * n_links
    heads += targets.tolist()
    in_degree[targets] += 1

# random links, in any direction
extra = rng.integers(0, n_pages, size=(1_000, 2))
links = pd.DataFrame(
    {
        "tail": np.concatenate((tails, extra[:, 0])),
        "head": np.concatenate((heads, extra[:, 1])),
    }
)
links = links[links["tail"] != links["head"]]
links.shape
```

% output

    (6989, 2)

% /output

## Structure

The graph is weakly connected: all the pages are connected when the link directions are ignored.

```python
from edsger import ConnectedComponents

cc = ConnectedComponents(links)
cc.run(connection="weak")
cc.n_components
```

% output

    1

% /output

```python
strong = cc.run(connection="strong")
cc.n_components, int(np.bincount(strong).max())  # components, size of the largest
```

% output

    (1279, 722)

% /output

Following the links, the largest strongly connected component has 722 pages, in which every page can be reached from every other one. The 1278 other components are single pages, which are on no cycle of links.

## PageRank

`Centrality` builds the graph once. `pagerank` returns the stationary distribution of a random surfer, who follows a link with probability 0.85 and otherwise jumps to a random page.

```python
import polars as pl

from edsger import Centrality

centrality = Centrality(links)
pagerank = centrality.pagerank(output="polars")
pagerank.sort("pagerank", descending=True).head()
```

% output

    shape: (5, 2)
    ┌────────────┬──────────┐
    │ vertex_idx ┆ pagerank │
    │ ---        ┆ ---      │
    │ i64        ┆ f64      │
    ╞════════════╪══════════╡
    │ 1          ┆ 0.06831  │
    │ 2          ┆ 0.060563 │
    │ 3          ┆ 0.059687 │
    │ 403        ┆ 0.051665 │
    │ 0          ┆ 0.050606 │
    └────────────┴──────────┘

% /output

The PageRank is related to the number of incoming links, but not equal to it: a link from an important page counts more. Joined with the in-degrees:

```python
in_links = (
    pl.from_pandas(links)
    .group_by("head")
    .len()
    .rename({"head": "vertex_idx", "len": "in_links"})
)
ranks = pagerank.join(in_links, on="vertex_idx")
ranks.select(pl.corr("pagerank", "in_links", method="spearman").round(3))
```

% output

    shape: (1, 1)
    ┌──────────┐
    │ pagerank │
    │ ---      │
    │ f64      │
    ╞══════════╡
    │ 0.867    │
    └──────────┘

% /output

With a `personalization`, the surfer always jumps back to the given pages: the result ranks the pages by their proximity to them.

```python
personalized = centrality.pagerank(personalization={1_500: 1.0}, output="polars")
personalized.sort("pagerank", descending=True).head(3)
```

% output

    shape: (3, 2)
    ┌────────────┬──────────┐
    │ vertex_idx ┆ pagerank │
    │ ---        ┆ ---      │
    │ i64        ┆ f64      │
    ╞════════════╪══════════╡
    │ 1500       ┆ 0.227902 │
    │ 2          ┆ 0.111829 │
    │ 403        ┆ 0.095054 │
    └────────────┴──────────┘

% /output

## Hubs and authorities

HITS gives two scores per page: authorities are pages pointed to by good hubs, and hubs are pages that point to good authorities. The authorities are close to the pages with the highest PageRank; the hubs are the pages that link to several of them.

```python
hubs, authorities = centrality.hits(max_iter=1_000, output="polars")
top = authorities.sort("authority", descending=True).head(5)["vertex_idx"].to_list()
top, hubs.sort("hub", descending=True).head(5)["vertex_idx"].to_list()
```

% output

    ([1, 3, 2, 0, 10], [41, 390, 7, 9, 6])

% /output

`katz(alpha)` is another measure of influence, which counts the walks that end at each page, a walk of length k counting for `alpha` to the power k.
