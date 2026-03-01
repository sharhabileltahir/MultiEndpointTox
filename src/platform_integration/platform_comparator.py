"""Compare predictions with external platforms (pkCSM, ProfHex, ProTox-II)."""
from loguru import logger

class PlatformComparator:
    def __init__(self, config): self.config = config
    def compare_all(self):
        logger.info("Platform comparison - implement after model training")
        return {}
