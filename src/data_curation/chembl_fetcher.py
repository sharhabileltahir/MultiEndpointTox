"""
ChEMBL Data Fetcher - Fetches bioactivity data for toxicity endpoints.
"""

import pandas as pd
from pathlib import Path
from loguru import logger

try:
    from chembl_webresource_client.new_client import new_client
except ImportError:
    logger.warning("chembl_webresource_client not installed. Run: pip install chembl_webresource_client")


class ChEMBLFetcher:

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_endpoint(self, endpoint):
        endpoint_config = self.config["data_curation"][endpoint]
        if endpoint == "herg":
            return self._fetch_herg(endpoint_config)
        elif endpoint == "hepatotox":
            return self._fetch_hepatotox(endpoint_config)
        elif endpoint == "nephrotox":
            return self._fetch_nephrotox(endpoint_config)
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")

    def _fetch_herg(self, cfg):
        logger.info(f"Querying ChEMBL for target: {cfg['target_chembl_id']}")
        activity = new_client.activity
        results = activity.filter(
            target_chembl_id=cfg["target_chembl_id"],
            standard_type__in=cfg["activity_types"],
            pchembl_value__gte=cfg.get("min_pchembl", 4.0),
        )
        df = pd.DataFrame.from_records(results)
        logger.info(f"Raw hERG records fetched: {len(df)}")

        output_path = self.raw_dir / "herg" / "herg_chembl_raw.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Raw data saved to {output_path}")
        return df

    def _fetch_hepatotox(self, cfg):
        logger.info("Querying ChEMBL for hepatotoxicity assays...")
        assay = new_client.assay
        all_records = []
        for keyword in cfg["target_keywords"]:
            results = assay.filter(description__icontains=keyword)
            assay_ids = [r["assay_chembl_id"] for r in results]
            logger.info(f"  Keyword '{keyword}': {len(assay_ids)} assays found")
            activity = new_client.activity
            for assay_id in assay_ids[:50]:
                try:
                    acts = activity.filter(assay_chembl_id=assay_id)
                    all_records.extend(list(acts))
                except Exception as e:
                    logger.debug(f"  Skipping {assay_id}: {e}")

        df = pd.DataFrame.from_records(all_records)
        logger.info(f"Raw hepatotoxicity records: {len(df)}")
        output_path = self.raw_dir / "hepatotox" / "hepatotox_chembl_raw.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    def _fetch_nephrotox(self, cfg):
        logger.info("Querying ChEMBL for nephrotoxicity assays...")
        assay = new_client.assay
        all_records = []
        for keyword in cfg["target_keywords"]:
            results = assay.filter(description__icontains=keyword)
            assay_ids = [r["assay_chembl_id"] for r in results]
            logger.info(f"  Keyword '{keyword}': {len(assay_ids)} assays found")
            activity = new_client.activity
            for assay_id in assay_ids[:50]:
                try:
                    acts = activity.filter(assay_chembl_id=assay_id)
                    all_records.extend(list(acts))
                except Exception as e:
                    logger.debug(f"  Skipping {assay_id}: {e}")

        df = pd.DataFrame.from_records(all_records)
        logger.info(f"Raw nephrotoxicity records: {len(df)}")
        output_path = self.raw_dir / "nephrotox" / "nephrotox_chembl_raw.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df
