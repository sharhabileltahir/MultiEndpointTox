"""Fetch skin sensitization data from ChEMBL or curated datasets."""

import pandas as pd
from pathlib import Path
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from chembl_webresource_client.new_client import new_client
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# Curated skin sensitization data from literature and regulatory databases
# Sources: LLNA data, ICCVAM validation studies, REACH database
SKIN_SENSITIZERS = [
    # Strong sensitizers
    "2,4-Dinitrochlorobenzene", "Oxazolone", "Glutaraldehyde", "Formaldehyde",
    "Glyoxal", "Methyldibromo glutaronitrile", "p-Phenylenediamine",
    "Tetramethylthiuram disulfide", "Mercaptobenzothiazole",
    # Moderate sensitizers
    "Isoeugenol", "Eugenol", "Cinnamaldehyde", "Cinnamic alcohol",
    "Citral", "Geraniol", "Hydroxycitronellal", "Coumarin",
    "Diethyl maleate", "2-Mercaptobenzothiazole", "Methylisothiazolinone",
    "Chloromethylisothiazolinone", "Kathon CG", "Imidazolidinyl urea",
    "Diazolidinyl urea", "DMDM hydantoin", "Bronopol",
    # Fragrances that sensitize
    "Linalool", "Limonene", "Alpha-pinene", "Benzyl alcohol",
    "Benzyl benzoate", "Benzyl cinnamate", "Amyl cinnamal",
    "Lyral", "Lilial", "HICC",
    # Preservatives/chemicals
    "Nickel sulfate", "Cobalt chloride", "Potassium dichromate",
    "Methylchloroisothiazolinone", "Phenyl glycidyl ether",
    "Epoxy resin", "Bisphenol A diglycidyl ether",
    # Plant extracts/naturals
    "Peru balsam", "Rosin", "Propolis", "Tea tree oil",
    "Turpentine", "Colophony",
    # Dyes and colorants
    "Disperse Blue 106", "Disperse Orange 3", "Basic Red 46",
    # Pharmaceuticals with sensitizing potential
    "Neomycin", "Bacitracin", "Benzocaine", "Procaine",
    "Chlorhexidine", "Povidone-iodine",
]

NON_SENSITIZERS = [
    # Common non-sensitizing compounds (LLNA negative)
    "Glycerol", "Propylene glycol", "Polyethylene glycol",
    "Sorbitol", "Mannitol", "Xylitol",
    "Sodium chloride", "Sodium bicarbonate", "Citric acid",
    "Lactic acid", "Malic acid", "Tartaric acid", "Succinic acid",
    "Glucose", "Sucrose", "Lactose", "Maltose", "Fructose",
    "Ethanol", "Isopropanol", "1-Propanol", "2-Propanol",
    "Acetone", "Dimethyl sulfoxide", "N-Methylpyrrolidone",
    "Petrolatum", "Mineral oil", "Paraffin", "Squalene",
    "Oleic acid", "Stearic acid", "Palmitic acid", "Myristic acid",
    "Caprylic acid", "Capric acid", "Lauric acid",
    "Cetyl alcohol", "Stearyl alcohol", "Cetostearyl alcohol",
    "Sodium lauryl sulfate", "Sodium laureth sulfate",
    "Cocamidopropyl betaine", "Decyl glucoside",
    "Methylparaben", "Ethylparaben", "Propylparaben", "Butylparaben",
    "Phenoxyethanol", "Sodium benzoate", "Potassium sorbate",
    "Tocopherol", "Ascorbic acid", "Retinol", "Niacinamide",
    "Panthenol", "Allantoin", "Urea", "Hyaluronic acid",
    "Caffeine", "Menthol", "Camphor", "Thymol",
    "Salicylic acid", "Benzoic acid", "Sorbic acid",
    "Titanium dioxide", "Zinc oxide", "Iron oxide",
    "Talc", "Kaolin", "Bentonite", "Silica",
    "Cyclomethicone", "Dimethicone", "Cyclohexasiloxane",
    "Isononyl isononanoate", "Ethylhexyl palmitate",
    "Caprylic capric triglyceride",
    "Water", "Sodium hydroxide", "Triethanolamine",
]


class SkinSensitizationFetcher:
    """
    Fetch skin sensitization data from ChEMBL or curated datasets.

    Skin sensitization (allergic contact dermatitis) is assessed via:
    - LLNA (Local Lymph Node Assay) - gold standard in vivo
    - Human patch tests
    - In vitro methods (KeratinoSens, h-CLAT, DPRA)

    Label: 1 = Sensitizer, 0 = Non-sensitizer
    """

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"]) / "skin_sens"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self):
        """
        Fetch skin sensitization data.

        Returns:
            pd.DataFrame: Dataset with SMILES and sensitization labels
        """
        logger.info("Fetching skin sensitization dataset...")

        csv_path = self.raw_dir / "skin_sens_curated_raw.csv"

        if csv_path.exists():
            logger.info(f"Using cached skin sensitization dataset: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            # Try ChEMBL first
            df = self._fetch_from_chembl()

            if len(df) < 100:
                logger.info("ChEMBL data insufficient, using curated dataset...")
                df = self._create_curated_dataset()

            df.to_csv(csv_path, index=False)
            logger.info(f"Saved skin sensitization data to {csv_path}")

        logger.info(f"Skin sensitization dataset: {len(df)} compounds")
        return df

    def _fetch_from_chembl(self):
        """Fetch skin sensitization data from ChEMBL."""
        logger.info("Querying ChEMBL for skin sensitization assays...")

        try:
            assay = new_client.assay
            activity = new_client.activity

            # Search for relevant assays
            keywords = ["skin sensitization", "LLNA", "local lymph node",
                       "contact dermatitis", "allergic contact", "KeratinoSens",
                       "h-CLAT", "DPRA"]
            all_records = []

            for keyword in keywords:
                results = assay.filter(description__icontains=keyword)
                assay_ids = [r["assay_chembl_id"] for r in results]
                logger.info(f"  Keyword '{keyword}': {len(assay_ids)} assays")

                for assay_id in assay_ids[:30]:
                    try:
                        acts = activity.filter(assay_chembl_id=assay_id)
                        for act in acts:
                            if act.get("canonical_smiles"):
                                all_records.append(act)
                    except Exception as e:
                        logger.debug(f"  Skipping {assay_id}: {e}")

            if not all_records:
                return pd.DataFrame()

            df = pd.DataFrame.from_records(all_records)
            df = self._process_chembl_skin_sens(df)

            return df

        except Exception as e:
            logger.warning(f"ChEMBL fetch failed: {e}")
            return pd.DataFrame()

    def _process_chembl_skin_sens(self, df):
        """Process ChEMBL data to extract sensitization labels."""
        if df.empty:
            return df

        df = df[df["canonical_smiles"].notna()].copy()

        def get_sens_label(row):
            comment = str(row.get("activity_comment", "")).lower()

            # Check for positive indicators
            if any(x in comment for x in ["sensitizer", "positive", "active", "allergenic"]):
                return 1
            if any(x in comment for x in ["non-sensitizer", "negative", "inactive", "not sensitizing"]):
                return 0
            return None

        df["activity_label"] = df.apply(get_sens_label, axis=1)
        df = df[df["activity_label"].notna()].copy()
        df["activity_label"] = df["activity_label"].astype(int)

        df["activity_comment"] = df["activity_label"].apply(
            lambda x: "Sensitizer" if x == 1 else "Non-sensitizer"
        )
        df["task_type"] = "classification"

        keep_cols = ["canonical_smiles", "molecule_chembl_id", "activity_label",
                     "activity_comment", "task_type"]
        existing_cols = [c for c in keep_cols if c in df.columns]
        df = df[existing_cols].drop_duplicates(subset=["canonical_smiles"])

        logger.info(f"Processed ChEMBL skin sensitization data: {len(df)} compounds")
        return df

    def _create_curated_dataset(self):
        """Create dataset from curated sensitization lists."""
        logger.info(f"Building curated skin sens dataset: {len(SKIN_SENSITIZERS)} sensitizers, {len(NON_SENSITIZERS)} non-sensitizers")

        records = []

        for compound in SKIN_SENSITIZERS:
            records.append({
                "drug_name": compound,
                "activity_label": 1,
                "activity_comment": "Sensitizer"
            })

        for compound in NON_SENSITIZERS:
            records.append({
                "drug_name": compound,
                "activity_label": 0,
                "activity_comment": "Non-sensitizer"
            })

        df = pd.DataFrame(records)

        logger.info("Fetching SMILES from ChEMBL...")
        df = self._add_smiles_from_chembl(df)

        return df

    def _add_smiles_from_chembl(self, df):
        """Look up SMILES structures from ChEMBL using compound names."""
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, skipping SMILES lookup")
            df["canonical_smiles"] = None
            df["molecule_chembl_id"] = None
            return df

        molecule = new_client.molecule

        smiles_list = []
        chembl_ids = []

        for name in df["drug_name"]:
            smiles = None
            chembl_id = None

            try:
                results = molecule.filter(
                    molecule_synonyms__molecule_synonym__iexact=name
                ).only(["molecule_chembl_id", "molecule_structures"])

                if results:
                    mol = results[0]
                    chembl_id = mol.get("molecule_chembl_id")
                    structures = mol.get("molecule_structures")
                    if structures:
                        smiles = structures.get("canonical_smiles")

                if not smiles:
                    results = molecule.filter(
                        pref_name__iexact=name
                    ).only(["molecule_chembl_id", "molecule_structures"])

                    if results:
                        mol = results[0]
                        chembl_id = mol.get("molecule_chembl_id")
                        structures = mol.get("molecule_structures")
                        if structures:
                            smiles = structures.get("canonical_smiles")

            except Exception as e:
                logger.debug(f"ChEMBL lookup failed for {name}: {e}")

            smiles_list.append(smiles)
            chembl_ids.append(chembl_id)

        df["canonical_smiles"] = smiles_list
        df["molecule_chembl_id"] = chembl_ids

        n_before = len(df)
        df = df[df["canonical_smiles"].notna()].copy()
        n_after = len(df)

        logger.info(f"Found SMILES for {n_after}/{n_before} compounds ({100*n_after/n_before:.1f}%)")

        # Standardize SMILES
        std_smiles = []
        for smiles in df["canonical_smiles"]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    std_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                else:
                    std_smiles.append(smiles)
            except:
                std_smiles.append(smiles)

        df["std_smiles"] = std_smiles
        df["task_type"] = "classification"

        n_pos = df["activity_label"].sum()
        n_neg = len(df) - n_pos
        logger.info(f"Skin sensitization dataset: {n_pos} sensitizers, {n_neg} non-sensitizers")

        return df
