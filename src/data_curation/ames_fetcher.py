"""Fetch Ames mutagenicity data from ChEMBL or curated datasets."""

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


# Curated Ames mutagenicity data from literature
# Sources: Hansen et al. (2009), Kazius et al. (2005), Benchmark datasets
AMES_POSITIVE = [
    # Known mutagens - aromatic amines
    "2-Aminofluorene", "4-Aminobiphenyl", "Benzidine", "2-Naphthylamine",
    "4-Nitroquinoline N-oxide", "2-Acetylaminofluorene",
    # Nitrosamines
    "N-Nitrosodimethylamine", "N-Nitrosodiethylamine", "N-Nitrosomorpholine",
    "N-Nitrosopiperidine", "N-Nitrosopyrrolidine",
    # PAHs
    "Benzo[a]pyrene", "7,12-Dimethylbenz[a]anthracene", "3-Methylcholanthrene",
    "Dibenz[a,h]anthracene",
    # Alkylating agents
    "Ethyl methanesulfonate", "Methyl methanesulfonate", "Ethylnitrosourea",
    "Methylnitrosourea", "Cyclophosphamide",
    # Heterocyclic amines
    "2-Amino-3-methylimidazo[4,5-f]quinoline", "2-Amino-1-methyl-6-phenylimidazo[4,5-b]pyridine",
    # Other mutagens
    "Aflatoxin B1", "Mitomycin C", "Azathioprine", "Chlorambucil",
    "Melphalan", "Busulfan", "Thiotepa", "Procarbazine",
    "Dacarbazine", "Temozolomide", "Carmustine", "Lomustine",
    "Streptozotocin", "Actinomycin D", "Daunorubicin", "Doxorubicin",
    "Bleomycin", "Etoposide", "Teniposide", "Topotecan",
    # Aromatic nitro compounds
    "2-Nitrofluorene", "4-Nitrophenol", "2,4-Dinitrotoluene", "2,6-Dinitrotoluene",
    # Additional known mutagens
    "Quinoline", "Acridine", "9-Aminoacridine", "Proflavine",
    "Acridine orange", "Ethidium bromide", "Safrole", "Estragole",
    "Colchicine", "Podophyllotoxin", "Vinblastine", "Vincristine",
]

AMES_NEGATIVE = [
    # Common non-mutagenic drugs
    "Acetaminophen", "Ibuprofen", "Aspirin", "Naproxen", "Caffeine",
    "Diphenhydramine", "Loratadine", "Cetirizine", "Fexofenadine",
    "Omeprazole", "Esomeprazole", "Pantoprazole", "Lansoprazole", "Ranitidine",
    "Metformin", "Glipizide", "Glyburide", "Pioglitazone", "Sitagliptin",
    "Atorvastatin", "Simvastatin", "Pravastatin", "Lovastatin", "Rosuvastatin",
    "Amlodipine", "Nifedipine", "Diltiazem", "Verapamil", "Felodipine",
    "Metoprolol", "Atenolol", "Propranolol", "Carvedilol", "Bisoprolol", "Nadolol",
    "Lisinopril", "Enalapril", "Captopril", "Ramipril", "Quinapril", "Fosinopril",
    "Losartan", "Valsartan", "Irbesartan", "Candesartan", "Olmesartan", "Telmisartan",
    "Sertraline", "Fluoxetine", "Paroxetine", "Citalopram", "Escitalopram", "Fluvoxamine",
    "Venlafaxine", "Duloxetine", "Bupropion", "Mirtazapine", "Trazodone",
    "Alprazolam", "Lorazepam", "Diazepam", "Clonazepam", "Oxazepam", "Temazepam",
    "Gabapentin", "Pregabalin", "Lamotrigine", "Levetiracetam", "Topiramate",
    "Amoxicillin", "Ampicillin", "Penicillin V", "Cephalexin", "Cefuroxime",
    "Azithromycin", "Clarithromycin", "Erythromycin", "Doxycycline", "Minocycline",
    "Ciprofloxacin", "Levofloxacin", "Moxifloxacin", "Ofloxacin",
    "Prednisone", "Prednisolone", "Dexamethasone", "Hydrocortisone", "Methylprednisolone",
    "Montelukast", "Zafirlukast", "Fluticasone", "Budesonide", "Beclomethasone",
    "Albuterol", "Salmeterol", "Formoterol", "Tiotropium", "Ipratropium",
    "Warfarin", "Clopidogrel", "Rivaroxaban", "Apixaban", "Dabigatran", "Enoxaparin",
    "Levothyroxine", "Liothyronine", "Methimazole", "Propylthiouracil",
    "Finasteride", "Tamsulosin", "Dutasteride", "Sildenafil", "Tadalafil",
    "Methotrexate", "Sulfasalazine", "Hydroxychloroquine", "Leflunomide",
]


class AmesFetcher:
    """
    Fetch Ames mutagenicity data from ChEMBL or curated datasets.

    The Ames test is a bacterial reverse mutation assay that detects
    mutagenic compounds. Positive = mutagenic, Negative = non-mutagenic.
    """

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"]) / "ames"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self):
        """
        Fetch Ames mutagenicity data.

        Returns:
            pd.DataFrame: Dataset with SMILES and mutagenicity labels
        """
        logger.info("Fetching Ames mutagenicity dataset...")

        csv_path = self.raw_dir / "ames_curated_raw.csv"

        if csv_path.exists():
            logger.info(f"Using cached Ames dataset: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            # Try ChEMBL first
            df = self._fetch_from_chembl()

            if len(df) < 100:
                logger.info("ChEMBL data insufficient, using curated dataset...")
                df = self._create_curated_dataset()

            df.to_csv(csv_path, index=False)
            logger.info(f"Saved Ames data to {csv_path}")

        logger.info(f"Ames mutagenicity dataset: {len(df)} compounds")
        return df

    def _fetch_from_chembl(self):
        """Fetch Ames test data from ChEMBL."""
        logger.info("Querying ChEMBL for Ames mutagenicity assays...")

        try:
            assay = new_client.assay
            activity = new_client.activity

            # Search for Ames test assays
            keywords = ["ames", "mutagenicity", "salmonella", "bacterial reverse mutation"]
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

            # Process activity comments to determine mutagenicity
            df = self._process_chembl_ames(df)

            return df

        except Exception as e:
            logger.warning(f"ChEMBL fetch failed: {e}")
            return pd.DataFrame()

    def _process_chembl_ames(self, df):
        """Process ChEMBL data to extract mutagenicity labels."""
        if df.empty:
            return df

        # Filter for compounds with SMILES
        df = df[df["canonical_smiles"].notna()].copy()

        # Determine activity from comments/values
        def get_ames_label(row):
            comment = str(row.get("activity_comment", "")).lower()
            value = row.get("standard_value")

            # Check for positive indicators
            if any(x in comment for x in ["positive", "mutagenic", "active", "genotoxic"]):
                return 1
            if any(x in comment for x in ["negative", "non-mutagenic", "inactive", "not mutagenic"]):
                return 0

            # If no clear label, skip
            return None

        df["activity_label"] = df.apply(get_ames_label, axis=1)
        df = df[df["activity_label"].notna()].copy()
        df["activity_label"] = df["activity_label"].astype(int)

        # Add standard columns
        df["activity_comment"] = df["activity_label"].apply(
            lambda x: "Mutagenic" if x == 1 else "Non-mutagenic"
        )
        df["task_type"] = "classification"

        # Keep relevant columns
        keep_cols = ["canonical_smiles", "molecule_chembl_id", "activity_label",
                     "activity_comment", "task_type"]
        existing_cols = [c for c in keep_cols if c in df.columns]
        df = df[existing_cols].drop_duplicates(subset=["canonical_smiles"])

        logger.info(f"Processed ChEMBL Ames data: {len(df)} compounds")
        return df

    def _create_curated_dataset(self):
        """Create dataset from curated mutagenicity lists with ChEMBL SMILES lookup."""
        logger.info(f"Building curated Ames dataset: {len(AMES_POSITIVE)} positive, {len(AMES_NEGATIVE)} negative")

        records = []

        # Add mutagenic compounds
        for compound in AMES_POSITIVE:
            records.append({
                "drug_name": compound,
                "activity_label": 1,
                "activity_comment": "Mutagenic"
            })

        # Add non-mutagenic compounds
        for compound in AMES_NEGATIVE:
            records.append({
                "drug_name": compound,
                "activity_label": 0,
                "activity_comment": "Non-mutagenic"
            })

        df = pd.DataFrame(records)

        # Get SMILES from ChEMBL
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
                # Search by synonym
                results = molecule.filter(
                    molecule_synonyms__molecule_synonym__iexact=name
                ).only(["molecule_chembl_id", "molecule_structures"])

                if results:
                    mol = results[0]
                    chembl_id = mol.get("molecule_chembl_id")
                    structures = mol.get("molecule_structures")
                    if structures:
                        smiles = structures.get("canonical_smiles")

                # Try pref_name if not found
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

        # Filter to compounds with SMILES
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

        # Log statistics
        n_pos = df["activity_label"].sum()
        n_neg = len(df) - n_pos
        logger.info(f"Ames dataset: {n_pos} mutagenic, {n_neg} non-mutagenic")

        return df
