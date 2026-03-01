"""Fetch cytotoxicity data from ChEMBL or curated datasets."""

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


# Curated cytotoxicity data - compounds with known cytotoxic profiles
# Sources: NCI60, ChEMBL cytotoxicity assays, literature
CYTOTOXIC_COMPOUNDS = [
    # Chemotherapy agents (inherently cytotoxic)
    "Doxorubicin", "Daunorubicin", "Epirubicin", "Idarubicin", "Mitoxantrone",
    "Paclitaxel", "Docetaxel", "Vincristine", "Vinblastine", "Vinorelbine",
    "Cisplatin", "Carboplatin", "Oxaliplatin", "Cyclophosphamide", "Ifosfamide",
    "Methotrexate", "Pemetrexed", "5-Fluorouracil", "Capecitabine", "Gemcitabine",
    "Etoposide", "Topotecan", "Irinotecan", "Camptothecin",
    "Bleomycin", "Actinomycin D", "Mitomycin C",
    "Bortezomib", "Carfilzomib", "Thalidomide", "Lenalidomide",
    # Cardiac glycosides (cytotoxic at high doses)
    "Digoxin", "Digitoxin", "Ouabain", "Oleandrin",
    # Antimicrobials with cytotoxicity
    "Amphotericin B", "Polymyxin B", "Colistin", "Vancomycin",
    # Immunosuppressants (cytotoxic mechanism)
    "Cyclosporine", "Tacrolimus", "Sirolimus", "Everolimus",
    "Azathioprine", "Mycophenolate mofetil", "Leflunomide",
    # Known cytotoxic chemicals
    "Arsenic trioxide", "Sodium arsenite", "Cadmium chloride",
    "Mercury chloride", "Lead acetate",
    "Potassium cyanide", "Sodium azide", "Sodium fluoride",
    "Chloroform", "Carbon tetrachloride", "Trichloroethylene",
    "Benzene", "Toluene", "Xylene",
    "Formaldehyde", "Glutaraldehyde", "Acrolein",
    "Hydrogen peroxide", "Sodium hypochlorite",
    "Staurosporine", "Camptothecin", "Colchicine",
    # Hepatotoxins with cytotoxic potential
    "Acetaminophen", "Aflatoxin B1", "Alpha-amanitin",
    "Phalloidin", "Microcystin-LR",
    # Antineoplastic kinase inhibitors
    "Sorafenib", "Sunitinib", "Pazopanib", "Regorafenib",
    "Imatinib", "Dasatinib", "Nilotinib", "Ponatinib",
    "Gefitinib", "Erlotinib", "Afatinib", "Osimertinib",
    "Vemurafenib", "Dabrafenib", "Trametinib", "Cobimetinib",
]

NON_CYTOTOXIC_COMPOUNDS = [
    # Common safe drugs (low cytotoxicity at therapeutic doses)
    "Aspirin", "Ibuprofen", "Naproxen", "Celecoxib",
    "Omeprazole", "Esomeprazole", "Pantoprazole", "Lansoprazole",
    "Ranitidine", "Famotidine", "Cimetidine",
    "Metformin", "Glipizide", "Glyburide", "Sitagliptin", "Saxagliptin",
    "Atorvastatin", "Simvastatin", "Pravastatin", "Rosuvastatin", "Lovastatin",
    "Amlodipine", "Nifedipine", "Diltiazem", "Verapamil",
    "Metoprolol", "Atenolol", "Propranolol", "Carvedilol", "Bisoprolol",
    "Lisinopril", "Enalapril", "Captopril", "Ramipril",
    "Losartan", "Valsartan", "Irbesartan", "Candesartan",
    "Hydrochlorothiazide", "Furosemide", "Spironolactone",
    "Sertraline", "Fluoxetine", "Paroxetine", "Citalopram", "Escitalopram",
    "Venlafaxine", "Duloxetine", "Bupropion", "Mirtazapine",
    "Alprazolam", "Lorazepam", "Diazepam", "Clonazepam",
    "Gabapentin", "Pregabalin", "Lamotrigine", "Levetiracetam",
    "Amoxicillin", "Ampicillin", "Penicillin V", "Cephalexin",
    "Azithromycin", "Clarithromycin", "Erythromycin",
    "Ciprofloxacin", "Levofloxacin", "Moxifloxacin",
    "Doxycycline", "Minocycline", "Tetracycline",
    "Prednisone", "Prednisolone", "Dexamethasone", "Hydrocortisone",
    "Montelukast", "Zafirlukast", "Fluticasone", "Budesonide",
    "Albuterol", "Salmeterol", "Formoterol", "Tiotropium",
    "Warfarin", "Clopidogrel", "Rivaroxaban", "Apixaban",
    "Levothyroxine", "Liothyronine", "Methimazole",
    "Finasteride", "Tamsulosin", "Sildenafil", "Tadalafil",
    "Loratadine", "Cetirizine", "Fexofenadine", "Diphenhydramine",
    "Caffeine", "Theophylline", "Nicotine",
    "Melatonin", "Vitamin D", "Vitamin B12", "Folic acid",
    "Biotin", "Thiamine", "Riboflavin", "Niacin", "Pantothenic acid",
]


class CytotoxFetcher:
    """
    Fetch cytotoxicity data from ChEMBL or curated datasets.

    Cytotoxicity is assessed via:
    - MTT assay
    - LDH release
    - ATP content
    - Cell viability assays

    Label: 1 = Cytotoxic, 0 = Non-cytotoxic
    """

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"]) / "cytotox"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self):
        """
        Fetch cytotoxicity data.

        Returns:
            pd.DataFrame: Dataset with SMILES and cytotoxicity labels
        """
        logger.info("Fetching cytotoxicity dataset...")

        csv_path = self.raw_dir / "cytotox_curated_raw.csv"

        if csv_path.exists():
            logger.info(f"Using cached cytotoxicity dataset: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            # Try ChEMBL first
            df = self._fetch_from_chembl()

            if len(df) < 100:
                logger.info("ChEMBL data insufficient, using curated dataset...")
                df = self._create_curated_dataset()

            df.to_csv(csv_path, index=False)
            logger.info(f"Saved cytotoxicity data to {csv_path}")

        logger.info(f"Cytotoxicity dataset: {len(df)} compounds")
        return df

    def _fetch_from_chembl(self):
        """Fetch cytotoxicity data from ChEMBL."""
        logger.info("Querying ChEMBL for cytotoxicity assays...")

        try:
            assay = new_client.assay
            activity = new_client.activity

            keywords = ["cytotoxicity", "cytotoxic", "cell viability",
                       "MTT assay", "LDH release", "cell death"]
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
            df = self._process_chembl_cytotox(df)

            return df

        except Exception as e:
            logger.warning(f"ChEMBL fetch failed: {e}")
            return pd.DataFrame()

    def _process_chembl_cytotox(self, df):
        """Process ChEMBL data to extract cytotoxicity labels."""
        if df.empty:
            return df

        df = df[df["canonical_smiles"].notna()].copy()

        def get_cytotox_label(row):
            comment = str(row.get("activity_comment", "")).lower()
            value = row.get("standard_value")
            units = str(row.get("standard_units", "")).lower()

            # Check comments
            if any(x in comment for x in ["cytotoxic", "toxic", "active", "cell death"]):
                return 1
            if any(x in comment for x in ["non-cytotoxic", "non-toxic", "inactive", "no effect"]):
                return 0

            # Check IC50/EC50 values (< 10 uM = cytotoxic)
            if value is not None and units in ["um", "µm", "micromolar"]:
                try:
                    if float(value) < 10:
                        return 1
                    else:
                        return 0
                except:
                    pass

            return None

        df["activity_label"] = df.apply(get_cytotox_label, axis=1)
        df = df[df["activity_label"].notna()].copy()
        df["activity_label"] = df["activity_label"].astype(int)

        df["activity_comment"] = df["activity_label"].apply(
            lambda x: "Cytotoxic" if x == 1 else "Non-cytotoxic"
        )
        df["task_type"] = "classification"

        keep_cols = ["canonical_smiles", "molecule_chembl_id", "activity_label",
                     "activity_comment", "task_type"]
        existing_cols = [c for c in keep_cols if c in df.columns]
        df = df[existing_cols].drop_duplicates(subset=["canonical_smiles"])

        logger.info(f"Processed ChEMBL cytotoxicity data: {len(df)} compounds")
        return df

    def _create_curated_dataset(self):
        """Create dataset from curated cytotoxicity lists."""
        logger.info(f"Building curated cytotox dataset: {len(CYTOTOXIC_COMPOUNDS)} cytotoxic, {len(NON_CYTOTOXIC_COMPOUNDS)} non-cytotoxic")

        records = []

        for compound in CYTOTOXIC_COMPOUNDS:
            records.append({
                "drug_name": compound,
                "activity_label": 1,
                "activity_comment": "Cytotoxic"
            })

        for compound in NON_CYTOTOXIC_COMPOUNDS:
            records.append({
                "drug_name": compound,
                "activity_label": 0,
                "activity_comment": "Non-cytotoxic"
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
        logger.info(f"Cytotoxicity dataset: {n_pos} cytotoxic, {n_neg} non-cytotoxic")

        return df
