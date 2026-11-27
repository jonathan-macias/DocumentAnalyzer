"""StateGraph construction.

START → ingest → parse ─┬─(scanned pages)→ vision_ocr ─┐
                        └──────────────────────────────┴→ classify_category
classify_category → classify_type → validate ─┬─ "retrieval" → index → extract
                                              ├─ "full" ─────────────→ extract
                                              └─ "skip" ─────────────→ finalize
extract → postprocess → finalize → END
Any node that sets state["error"] → finalize.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import AnalyzeNodes
from app.graph.state import AnalyzeState

FINALIZE = "finalize"


def ok_or_finalize(next_node: str):
    """Generic router: error → finalize; otherwise → next_node."""

    def route(state: AnalyzeState) -> str:
        return FINALIZE if state.get("error") else next_node

    route.__name__ = f"to_{next_node}_or_finalize"
    return route


def route_after_parse(state: AnalyzeState) -> str:
    if state.get("error"):
        return FINALIZE
    needs_ocr = any(p.get("needs_ocr") for p in state.get("pages", []))
    return "vision_ocr" if needs_ocr else "classify_category"


def route_after_validate(state: AnalyzeState) -> str:
    if state.get("error"):
        return FINALIZE
    return {"retrieval": "index", "full": "extract"}.get(state.get("extraction_path"), FINALIZE)


def build_graph(nodes: AnalyzeNodes, checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(AnalyzeState)

    g.add_node("ingest", nodes.ingest)
    g.add_node("parse", nodes.parse)
    g.add_node("vision_ocr", nodes.vision_ocr)
    g.add_node("classify_category", nodes.classify_category)
    g.add_node("classify_type", nodes.classify_type)
    g.add_node("validate", nodes.validate)
    g.add_node("index", nodes.index)
    g.add_node("extract", nodes.extract)
    g.add_node("postprocess", nodes.postprocess)
    g.add_node(FINALIZE, nodes.finalize)

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", ok_or_finalize("parse"), ["parse", FINALIZE])
    g.add_conditional_edges(
        "parse", route_after_parse, ["vision_ocr", "classify_category", FINALIZE]
    )
    g.add_conditional_edges(
        "vision_ocr", ok_or_finalize("classify_category"), ["classify_category", FINALIZE]
    )
    g.add_conditional_edges(
        "classify_category", ok_or_finalize("classify_type"), ["classify_type", FINALIZE]
    )
    g.add_conditional_edges("classify_type", ok_or_finalize("validate"), ["validate", FINALIZE])
    g.add_conditional_edges("validate", route_after_validate, ["index", "extract", FINALIZE])
    g.add_conditional_edges("index", ok_or_finalize("extract"), ["extract", FINALIZE])
    g.add_conditional_edges("extract", ok_or_finalize("postprocess"), ["postprocess", FINALIZE])
    g.add_edge("postprocess", FINALIZE)
    g.add_edge(FINALIZE, END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
