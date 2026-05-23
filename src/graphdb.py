from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
from pyvis.network import Network

from .models import ClassifiedItem


def build_graph(items: List[ClassifiedItem]) -> nx.Graph:
    g = nx.Graph()

    g.add_node("ROOT", label="VOC/News Intelligence", type="root", size=34, title="전체 분석 루트")

    category_counts = Counter([it.category for it in items])
    for cat, cnt in category_counts.items():
        g.add_node(f"CAT::{cat}", label=cat, type="category", size=18 + cnt * 2, title=f"{cat}: {cnt}건")
        g.add_edge("ROOT", f"CAT::{cat}", weight=max(1, cnt), label=str(cnt))

    for it in items:
        item_node = f"ITEM::{it.id}"
        item_label = compact_label(it.title, 34)
        g.add_node(
            item_node,
            label=item_label,
            type="item",
            source=it.source,
            category=it.category,
            sentiment=it.sentiment,
            priority=it.priority,
            url=it.url,
            published=it.published,
            title=f"{it.title}\n\n카테고리: {it.category}\n감성: {it.sentiment}\n중요도: {it.priority}\n이유: {it.reason}",
            size=10 + it.priority * 3,
        )
        g.add_edge(f"CAT::{it.category}", item_node, weight=max(1, it.priority), label=str(it.priority))

        for kw in (it.keywords or [])[:5]:
            kw = str(kw).strip()
            if not kw:
                continue
            kw_node = f"KW::{kw}"
            if not g.has_node(kw_node):
                g.add_node(kw_node, label=kw, type="keyword", size=10, title=f"키워드: {kw}")
            else:
                g.nodes[kw_node]["size"] = min(32, g.nodes[kw_node].get("size", 10) + 1)
            g.add_edge(item_node, kw_node, weight=1)

        if it.source:
            src_node = f"SRC::{it.source}"
            if not g.has_node(src_node):
                g.add_node(src_node, label=compact_label(it.source, 22), type="source", size=12, title=f"출처: {it.source}")
            g.add_edge(src_node, item_node, weight=1)

    return g


def compact_label(text: str, limit: int = 30) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_pyvis_html(g: nx.Graph, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="760px", width="100%", bgcolor="#111827", font_color="#F9FAFB", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-52000, central_gravity=0.25, spring_length=155, spring_strength=0.03, damping=0.86)

    color_by_type = {
        "root": "#F59E0B",
        "category": "#60A5FA",
        "item": "#34D399",
        "keyword": "#F472B6",
        "source": "#A78BFA",
    }

    for node, attrs in g.nodes(data=True):
        ntype = attrs.get("type", "item")
        net.add_node(
            node,
            label=attrs.get("label", str(node)),
            title=attrs.get("title", attrs.get("label", str(node))),
            color=color_by_type.get(ntype, "#9CA3AF"),
            size=attrs.get("size", 12),
            shape="dot" if ntype != "item" else "ellipse",
        )

    for a, b, attrs in g.edges(data=True):
        net.add_edge(a, b, value=attrs.get("weight", 1), title=attrs.get("label", ""))

    net.set_options(
        """
        {
          "nodes": {"font": {"size": 15, "face": "Arial"}},
          "edges": {"color": {"color": "#6B7280"}, "smooth": {"type": "continuous"}},
          "physics": {"stabilization": {"iterations": 160}}
        }
        """
    )
    net.write_html(str(output_path), notebook=False)
    return output_path.read_text(encoding="utf-8")


def graph_to_json(g: nx.Graph) -> Dict:
    return {
        "nodes": [{"id": n, **attrs} for n, attrs in g.nodes(data=True)],
        "edges": [{"source": a, "target": b, **attrs} for a, b, attrs in g.edges(data=True)],
    }


def graph_to_cypher(items: List[ClassifiedItem]) -> str:
    """
    Neo4j Browser/Aura Console에서 실행 가능한 Cypher.
    대량 운영에서는 neo4j Python 드라이버의 write_to_neo4j 사용 권장.
    """
    lines = [
        "CREATE CONSTRAINT item_id IF NOT EXISTS FOR (i:Item) REQUIRE i.id IS UNIQUE;",
        "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;",
        "CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE;",
        "CREATE CONSTRAINT source_name IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE;",
        "",
    ]
    for it in items:
        data = {
            "id": it.id,
            "title": it.title,
            "text": it.text[:1200],
            "url": it.url,
            "published": it.published,
            "sentiment": it.sentiment,
            "priority": it.priority,
            "reason": it.reason,
        }
        lines.append(f"MERGE (i:Item {{id:{cypher_str(it.id)}}}) SET i += {cypher_map(data)};")
        lines.append(f"MERGE (c:Category {{name:{cypher_str(it.category)}}});")
        lines.append(f"MERGE (s:Source {{name:{cypher_str(it.source)}}});")
        lines.append(f"MATCH (i:Item {{id:{cypher_str(it.id)}}}), (c:Category {{name:{cypher_str(it.category)}}}) MERGE (i)-[:IN_CATEGORY]->(c);")
        lines.append(f"MATCH (i:Item {{id:{cypher_str(it.id)}}}), (s:Source {{name:{cypher_str(it.source)}}}) MERGE (i)-[:FROM_SOURCE]->(s);")
        for kw in (it.keywords or [])[:6]:
            lines.append(f"MERGE (k:Keyword {{name:{cypher_str(kw)}}});")
            lines.append(f"MATCH (i:Item {{id:{cypher_str(it.id)}}}), (k:Keyword {{name:{cypher_str(kw)}}}) MERGE (i)-[:HAS_KEYWORD]->(k);")
        lines.append("")
    return "\n".join(lines)


def write_to_neo4j(items: List[ClassifiedItem], uri: str, user: str, password: str) -> Tuple[bool, str]:
    if not (uri and user and password):
        return False, "Neo4j URI/USER/PASSWORD가 비어 있습니다."
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        return False, f"neo4j 패키지를 불러오지 못했습니다: {e}"

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("CREATE CONSTRAINT item_id IF NOT EXISTS FOR (i:Item) REQUIRE i.id IS UNIQUE")
            session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE")
            session.run("CREATE CONSTRAINT source_name IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE")

            for it in items:
                session.run(
                    """
                    MERGE (i:Item {id:$id})
                    SET i.title=$title, i.text=$text, i.url=$url, i.published=$published,
                        i.sentiment=$sentiment, i.priority=$priority, i.reason=$reason
                    MERGE (c:Category {name:$category})
                    MERGE (s:Source {name:$source})
                    MERGE (i)-[:IN_CATEGORY]->(c)
                    MERGE (i)-[:FROM_SOURCE]->(s)
                    """,
                    id=it.id,
                    title=it.title,
                    text=it.text[:3000],
                    url=it.url,
                    published=it.published,
                    sentiment=it.sentiment,
                    priority=it.priority,
                    reason=it.reason,
                    category=it.category,
                    source=it.source,
                )
                for kw in (it.keywords or [])[:6]:
                    session.run(
                        """
                        MATCH (i:Item {id:$id})
                        MERGE (k:Keyword {name:$kw})
                        MERGE (i)-[:HAS_KEYWORD]->(k)
                        """,
                        id=it.id,
                        kw=kw,
                    )
        driver.close()
        return True, f"Neo4j GraphDB에 {len(items)}개 아이템 저장 완료"
    except Exception as e:
        return False, f"Neo4j 저장 실패: {type(e).__name__}: {e}"


def cypher_str(value: str) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def cypher_map(d: Dict) -> str:
    parts = []
    for k, v in d.items():
        key = "`" + str(k).replace("`", "") + "`"
        if isinstance(v, (int, float)):
            val = str(v)
        else:
            val = cypher_str(v)
        parts.append(f"{key}: {val}")
    return "{" + ", ".join(parts) + "}"
