from langgraph.graph.state import CompiledStateGraph
from loguru import logger


def visualize_graph(graph: CompiledStateGraph):
    """Try to visualize the graph if possible."""
    try:
        from IPython.display import Image, display  # type: ignore
    except ImportError:
        logger.warning("IPython is not available. Cannot display graph visualization.")
        return
    try:
        img_data = graph.get_graph().draw_png()
        display(Image(img_data))
    except Exception as e:
        logger.error(f"Graph visualization failed (requires graphviz): {e}")
