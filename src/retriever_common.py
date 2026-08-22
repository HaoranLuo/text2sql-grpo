"""Shared utilities for the schema retriever line (P1-2).

Reference implementation: LitE-SQL schema_retriever
(tmp_idea_research/bird_gen_scan/code/LitE-SQL/schema_retriever), adapted from
column-level to TABLE-level schema items.

Contents:
- build_query_text / build_table_doc : query & schema-item text templates
- extract_related_tables              : gold-SQL -> related table set
                                         (sqlglot AST first, regex fallback;
                                          alias + case handled via dict match)
- last_token_pool                     : Qwen3-Embedding style pooling
- hard_negative_supcon_loss           : HN-SupCon loss, verbatim LitE-SQL logic
"""

import re
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F
import sqlglot
from sqlglot import exp

QUERY_INSTRUCTION = (
    "Instruct: Given a natural language question, retrieve the database table "
    "information needed to generate SQL.\n"
)


# --------------------------------------------------------------------------
# text templates
# --------------------------------------------------------------------------
def build_query_text(question: str, evidence: str = "") -> str:
    """Query side of the bi-encoder (LitE-SQL instruction format)."""
    text = f"{QUERY_INSTRUCTION}Query: {question}"
    if evidence:
        text += f" {evidence}"
    return text


def build_table_doc(
    table_name: str,
    columns: List[str],
    col_types: Optional[Dict[str, str]] = None,
    col_comments: Optional[Dict[str, str]] = None,
) -> str:
    """Schema-item (single table) text: name + column list (+types) + comments.

    Tag style follows LitE-SQL page_content (`<table>... <column>... <column
    description>...`) but with the whole column list inside one `<columns>` tag.
    """
    col_types = col_types or {}
    col_comments = col_comments or {}

    col_list = []
    for c in columns:
        t = col_types.get(c, "")
        col_list.append(f"{c}({t})" if t else str(c))
    parts = [f"<table>{table_name}</table>", "<columns>" + ", ".join(col_list) + "</columns>"]

    desc = "; ".join(f"{c}: {d}" for c, d in col_comments.items() if d and d.strip())
    if desc:
        parts.append(f"<column descriptions>{desc}</column descriptions>")
    return " ".join(parts)


# --------------------------------------------------------------------------
# gold SQL -> related tables
# --------------------------------------------------------------------------
def _tables_from_ast(parsed, db_tables_lower: Set[str]) -> Set[str]:
    """Table-level version of LitE-SQL get_related_tab_col (no column part)."""
    cte_tables: Set[str] = set()
    for cte in parsed.find_all(exp.CTE):
        for attr in ("alias", "this"):
            node = getattr(cte, attr, None)
            if node is not None:
                name = getattr(node, "alias_or_name", None) or getattr(node, "name", None)
                if name:
                    cte_tables.add(str(name).lower())

    table_alias_map: Dict[str, str] = {}
    for t in parsed.find_all(exp.Table):
        try:
            name = t.this.name
        except AttributeError:
            name = str(t.this).lower() if t.this is not None else ""
        name = str(name).lower()
        alias = str(t.alias_or_name).lower()
        if name and name not in cte_tables and alias not in cte_tables:
            table_alias_map[alias] = name

    related: Set[str] = set()
    # explicit FROM/JOIN tables
    for name in table_alias_map.values():
        if name in db_tables_lower:
            related.add(name)
    # tables referenced only via qualified columns (e.g. inside subqueries)
    for col in parsed.find_all(exp.Column):
        if col.table:
            ref = str(col.table).lower()
            orig = table_alias_map.get(ref, ref)
            if orig in db_tables_lower:
                related.add(orig)
    return related


_CLAUSE_BREAK = re.compile(
    r"\b(?:where|group\s+by|order\s+by|having|limit|offset|union|intersect|"
    r"except|on|join|left|right|inner|outer|full|cross|natural|using|;|\))\b",
    re.IGNORECASE,
)


def _tables_from_regex(sql_lower: str, db_tables_lower: Set[str]) -> Set[str]:
    """Fallback when sqlglot cannot parse: scan FROM/JOIN segments and match
    against the db's real table-name dictionary (handles aliases & case)."""
    related: Set[str] = set()
    for m in re.finditer(r"\b(?:from|join)\b", sql_lower):
        seg = sql_lower[m.end(): m.end() + 200]
        cut = _CLAUSE_BREAK.search(seg)
        if cut:
            seg = seg[: cut.start()]
        seg = re.sub(r"[`\"'\[\]]", " ", seg)
        seg = re.sub(r"\s+", " ", f" {seg} ")
        for name in db_tables_lower:
            if name in related:
                continue
            if " " in name:  # names with spaces: literal match inside segment
                if f" {name} " in seg:
                    related.add(name)
            else:
                for tok in re.split(r"[^a-z0-9_]+", seg):
                    if tok == name or tok.endswith("." + name):
                        related.add(name)
                        break
    return related


def extract_related_tables(
    sql: str,
    db_tables_lower: Set[str],
    dialect: str = "sqlite",
) -> Set[str]:
    """Related (lowercased) table names referenced by a gold SQL.

    db_tables_lower: real table names of the database (lowercased), used as the
    matching dictionary so aliases / case variants cannot leak in.
    """
    if not sql or not db_tables_lower:
        return set()
    sql_lower = (sql or "").lower()
    try:
        parsed = sqlglot.parse_one(sql_lower, read=dialect)
        if parsed is not None:
            related = _tables_from_ast(parsed, db_tables_lower)
            if related:
                return related
    except Exception:
        pass
    return _tables_from_regex(sql_lower, db_tables_lower)


# --------------------------------------------------------------------------
# pooling + loss (verbatim LitE-SQL semantics)
# --------------------------------------------------------------------------
def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """LitE-SQL last-token pooling with left-padding detection."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def hard_negative_supcon_loss(
    query_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding_list: List[torch.Tensor],
    temperature: float = 0.07,
    hard_negative_threshold: float = 0.1,
    too_hard_negative: bool = True,
) -> torch.Tensor:
    """HN-SupCon loss, logic copied verbatim from LitE-SQL
    schema_retriever/scripts/fine-tune.py::HardNegativeSuperConLoss.

    negative_embedding_list: list of per-sample tensors [n_i, dim] (n_i may vary).
    With too_hard_negative=True only negatives whose sim to the query is within
    `hard_negative_threshold` of the positive's sim enter the logsumexp; if none
    qualify the single hardest negative is used (never a zero gradient).
    """

    def _l2norm(x):
        return F.normalize(x, p=2, dim=-1)

    q = _l2norm(query_embedding)
    d_pos = _l2norm(positive_embedding)
    d_negs = [_l2norm(neg) for neg in negative_embedding_list]

    B, _ = q.shape

    sim_pos = (q * d_pos).sum(dim=-1)

    sim_neg_list = []
    for i in range(B):
        if too_hard_negative:
            sim_neg_i = (q[i] * d_negs[i]).sum(dim=-1)
            keep = sim_neg_i >= (sim_pos[i].detach() - hard_negative_threshold)
            if keep.any():
                sim_neg_list.append(torch.logsumexp(sim_neg_i[keep] / temperature, dim=0))
            else:
                sim_neg_list.append(torch.max(sim_neg_i) / temperature)
        else:
            sim_neg_i = (q[i] * d_negs[i] / temperature).sum(dim=-1)
            sim_neg_list.append(torch.logsumexp(sim_neg_i, dim=0))

    sim_negs = torch.stack(sim_neg_list)
    sim_pos = sim_pos / temperature
    z_stack = torch.stack([sim_pos, sim_negs], dim=0)
    return (torch.logsumexp(z_stack, dim=0) - sim_pos).mean()
