import logging
import sys
from selection.index_selection_evaluation import IndexSelection  # pragma: no cover

BASIC_FORMAT = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
logging.basicConfig(
    format=BASIC_FORMAT,
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("output.log"),
        logging.StreamHandler(),
    ],
)

index_selection = IndexSelection()  # pragma: no cover
index_selection.run()  # pragma: no cover
