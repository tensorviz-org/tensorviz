"""Named by the SDK, so its identity comes from the declaration."""
import tensorviz

from models.net import Net

tensorviz.experiment("wide-net")
tensorviz.node("dataset", "raw-logs")
tensorviz.node("transform", "tokenise")
tensorviz.edge("raw-logs", "tokenise", "feeds")


def build():
    return Net(hidden=64)


if __name__ == "__main__":
    build()
