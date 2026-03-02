"""Fetch reproductive/developmental toxicity data from ChEMBL or curated datasets."""

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


# Curated reproductive/developmental toxicity data from literature
# Sources: FDA pregnancy categories, DART studies, ToxCast, literature reviews
# Known teratogens and reproductive toxicants (Category X or known human teratogens)
REPRODUCTIVE_TOX_POSITIVE = [
    # Retinoids - well-documented teratogens
    "Isotretinoin", "Acitretin", "Etretinate", "Tretinoin", "Tazarotene",
    # Thalidomide and analogs
    "Thalidomide", "Lenalidomide", "Pomalidomide",
    # Anticonvulsants with teratogenic effects
    "Valproic acid", "Phenytoin", "Carbamazepine", "Phenobarbital",
    "Primidone", "Trimethadione", "Paramethadione",
    # Antineoplastics / cytotoxic agents
    "Methotrexate", "Cyclophosphamide", "Chlorambucil", "Busulfan",
    "Mercaptopurine", "Fluorouracil", "Cytarabine", "Doxorubicin",
    "Daunorubicin", "Bleomycin", "Vinblastine", "Vincristine",
    "Etoposide", "Ifosfamide", "Melphalan", "Procarbazine",
    # Hormones and hormone modulators
    "Diethylstilbestrol", "Danazol", "Finasteride", "Dutasteride",
    "Testosterone", "Methyltestosterone", "Fluoxymesterone",
    # ACE inhibitors and ARBs (fetotoxic)
    "Lisinopril", "Enalapril", "Captopril", "Ramipril", "Quinapril",
    "Losartan", "Valsartan", "Irbesartan", "Candesartan", "Olmesartan",
    # Anticoagulants
    "Warfarin", "Phenprocoumon", "Acenocoumarol",
    # Antifungals
    "Fluconazole", "Voriconazole", "Itraconazole",
    # Statins
    "Atorvastatin", "Simvastatin", "Lovastatin", "Pravastatin", "Rosuvastatin",
    # Antimicrobials with developmental toxicity
    "Ribavirin", "Ganciclovir", "Valganciclovir",
    # Immunosuppressants
    "Mycophenolate mofetil", "Leflunomide",
    # Other known teratogens
    "Misoprostol", "Lithium carbonate", "Bosentan", "Ambrisentan",
    "Ergotamine", "Methimazole", "Propylthiouracil",
    # Environmental/industrial reproductive toxicants
    "Lead acetate", "Methylmercury", "Cadmium chloride",
    # Additional reproductive toxicants from literature
    "Aminopterin", "Methyl ethyl ketone peroxide", "Ethylene glycol",
    "Benomyl", "Carbendazim", "Colchicine",
]

# Known safe compounds (Category A/B or extensive safe use data)
REPRODUCTIVE_TOX_NEGATIVE = [
    # Prenatal vitamins and supplements
    "Folic acid", "Pyridoxine", "Thiamine", "Riboflavin", "Cyanocobalamin",
    "Ascorbic acid", "Cholecalciferol", "Iron sulfate",
    # Safe analgesics
    "Acetaminophen",
    # Safe antibiotics (Category B)
    "Penicillin V", "Amoxicillin", "Ampicillin", "Cephalexin", "Cefuroxime",
    "Ceftriaxone", "Azithromycin", "Erythromycin", "Clindamycin", "Nitrofurantoin",
    # Safe antiemetics
    "Ondansetron", "Doxylamine", "Meclizine", "Dimenhydrinate",
    # Safe antihistamines
    "Loratadine", "Cetirizine", "Diphenhydramine", "Chlorpheniramine",
    # Safe GI medications
    "Ranitidine", "Famotidine", "Omeprazole", "Pantoprazole",
    "Sucralfate", "Calcium carbite",
    # Safe cardiovascular drugs
    "Methyldopa", "Labetalol", "Nifedipine", "Hydralazine",
    # Safe respiratory medications
    "Albuterol", "Budesonide", "Fluticasone", "Montelukast",
    # Safe thyroid medications (levothyroxine)
    "Levothyroxine",
    # Safe diabetes medications
    "Insulin", "Metformin",
    # Safe psychiatric medications (relatively)
    "Sertraline", "Fluoxetine",
    # Safe topical agents
    "Mupirocin", "Clotrimazole", "Miconazole", "Nystatin",
    # Common safe OTC drugs
    "Guaifenesin", "Dextromethorphan", "Pseudoephedrine",
    # Additional Category B drugs
    "Metoclopramide", "Prochlorperazine", "Promethazine",
    "Acyclovir", "Valacyclovir",
    "Ibuprofen", "Naproxen",  # Safe in 1st/2nd trimester
    # Natural compounds generally safe
    "Ginger extract", "Peppermint oil",
]


class ReproductiveToxFetcher:
    """
    Fetch reproductive/developmental toxicity data from ChEMBL or curated datasets.

    Reproductive toxicity includes:
    - Teratogenicity (structural birth defects)
    - Developmental toxicity (functional/growth effects)
    - Fertility effects
    - Embryotoxicity/fetotoxicity

    Classification:
    - Positive (1): Known reproductive toxicant/teratogen
    - Negative (0): No evidence of reproductive toxicity
    """

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"]) / "reproductive_tox"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self):
        """
        Fetch reproductive toxicity data.

        Returns:
            pd.DataFrame: Dataset with SMILES and toxicity labels
        """
        logger.info("Fetching reproductive toxicity dataset...")

        csv_path = self.raw_dir / "reproductive_tox_curated_raw.csv"

        if csv_path.exists():
            logger.info(f"Using cached reproductive toxicity dataset: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            # Try ChEMBL first
            df = self._fetch_from_chembl()

            if len(df) < 100:
                logger.info("ChEMBL data insufficient, using curated dataset...")
                df = self._create_curated_dataset()

            df.to_csv(csv_path, index=False)
            logger.info(f"Saved reproductive toxicity data to {csv_path}")

        logger.info(f"Reproductive toxicity dataset: {len(df)} compounds")
        return df

    def _fetch_from_chembl(self):
        """Fetch reproductive toxicity data from ChEMBL."""
        logger.info("Querying ChEMBL for reproductive toxicity assays...")

        try:
            assay = new_client.assay
            activity = new_client.activity

            # Search for reproductive/developmental toxicity assays
            keywords = [
                "reproductive toxicity", "developmental toxicity",
                "teratogenicity", "teratogenic", "embryotoxicity",
                "fertility", "DART"  # Developmental and Reproductive Toxicology
            ]
            all_records = []

            for keyword in keywords:
                try:
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
                except Exception as e:
                    logger.debug(f"  Keyword search failed for '{keyword}': {e}")

            if not all_records:
                return pd.DataFrame()

            df = pd.DataFrame.from_records(all_records)

            # Process activity comments to determine toxicity
            df = self._process_chembl_reproductive_tox(df)

            return df

        except Exception as e:
            logger.warning(f"ChEMBL fetch failed: {e}")
            return pd.DataFrame()

    def _process_chembl_reproductive_tox(self, df):
        """Process ChEMBL data to extract reproductive toxicity labels."""
        if df.empty:
            return df

        # Filter for compounds with SMILES
        df = df[df["canonical_smiles"].notna()].copy()

        # Determine activity from comments/values
        def get_repro_tox_label(row):
            comment = str(row.get("activity_comment", "")).lower()
            value = row.get("standard_value")

            # Check for positive indicators
            positive_terms = [
                "teratogenic", "embryotoxic", "fetotoxic", "reproductive toxicant",
                "developmental toxicity", "positive", "active", "toxic",
                "malformation", "birth defect", "fertility impaired"
            ]
            if any(term in comment for term in positive_terms):
                return 1

            # Check for negative indicators
            negative_terms = [
                "non-teratogenic", "not teratogenic", "negative", "inactive",
                "no effect", "safe", "no developmental", "no reproductive"
            ]
            if any(term in comment for term in negative_terms):
                return 0

            # If no clear label, skip
            return None

        df["activity_label"] = df.apply(get_repro_tox_label, axis=1)
        df = df[df["activity_label"].notna()].copy()
        df["activity_label"] = df["activity_label"].astype(int)

        # Add standard columns
        df["activity_comment"] = df["activity_label"].apply(
            lambda x: "Reproductive toxicant" if x == 1 else "Non-reproductive toxicant"
        )
        df["task_type"] = "classification"

        # Keep relevant columns
        keep_cols = ["canonical_smiles", "molecule_chembl_id", "activity_label",
                     "activity_comment", "task_type"]
        existing_cols = [c for c in keep_cols if c in df.columns]
        df = df[existing_cols].drop_duplicates(subset=["canonical_smiles"])

        logger.info(f"Processed ChEMBL reproductive toxicity data: {len(df)} compounds")
        return df

    def _create_curated_dataset(self):
        """Create dataset from curated toxicity lists with ChEMBL SMILES lookup."""
        logger.info(f"Building curated reproductive toxicity dataset: "
                    f"{len(REPRODUCTIVE_TOX_POSITIVE)} positive, "
                    f"{len(REPRODUCTIVE_TOX_NEGATIVE)} negative")

        records = []

        # Add toxic compounds
        for compound in REPRODUCTIVE_TOX_POSITIVE:
            records.append({
                "drug_name": compound,
                "activity_label": 1,
                "activity_comment": "Reproductive toxicant"
            })

        # Add non-toxic compounds
        for compound in REPRODUCTIVE_TOX_NEGATIVE:
            records.append({
                "drug_name": compound,
                "activity_label": 0,
                "activity_comment": "Non-reproductive toxicant"
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
        logger.info(f"Reproductive toxicity dataset: {n_pos} toxic, {n_neg} non-toxic")

        return df
