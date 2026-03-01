"""Fetch nephrotoxicity data from FDA DIRIL dataset or curated fallback."""

import pandas as pd
import requests
from pathlib import Path
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from chembl_webresource_client.new_client import new_client
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# Curated list of nephrotoxic drugs from literature and FDA labels
# Sources: DrugBank, FDA DIKI labels, PubMed literature reviews
NEPHROTOXIC_DRUGS = [
    # Aminoglycosides
    "Gentamicin", "Tobramycin", "Amikacin", "Streptomycin", "Neomycin", "Kanamycin",
    # NSAIDs
    "Ibuprofen", "Naproxen", "Indomethacin", "Piroxicam", "Ketorolac", "Diclofenac",
    "Meloxicam", "Celecoxib", "Sulindac", "Ketoprofen",
    # ACE inhibitors / ARBs
    "Lisinopril", "Enalapril", "Captopril", "Ramipril", "Losartan", "Valsartan",
    "Irbesartan", "Candesartan", "Olmesartan",
    # Contrast agents
    "Iohexol", "Iopamidol", "Iodixanol", "Ioversol",
    # Chemotherapy
    "Cisplatin", "Carboplatin", "Methotrexate", "Ifosfamide", "Cyclophosphamide",
    "Mitomycin", "Streptozocin",
    # Immunosuppressants
    "Cyclosporine", "Tacrolimus", "Sirolimus",
    # Antivirals
    "Acyclovir", "Cidofovir", "Foscarnet", "Adefovir", "Tenofovir", "Indinavir",
    # Antibiotics (other)
    "Vancomycin", "Amphotericin B", "Colistin", "Polymyxin B", "Rifampin",
    "Sulfamethoxazole", "Trimethoprim", "Penicillin G",
    # Diuretics
    "Furosemide", "Triamterene", "Hydrochlorothiazide",
    # Others
    "Lithium", "Phenytoin", "Allopurinol", "Probenecid", "Penicillamine",
    "Gold sodium thiomalate", "Zoledronic acid", "Pamidronate",
]

NON_NEPHROTOXIC_DRUGS = [
    # Common safe drugs (minimal nephrotoxicity at normal doses)
    "Acetaminophen", "Aspirin", "Caffeine", "Diphenhydramine", "Loratadine",
    "Cetirizine", "Famotidine", "Omeprazole", "Pantoprazole", "Esomeprazole",
    "Metformin", "Glipizide", "Pioglitazone", "Sitagliptin",
    "Atorvastatin", "Simvastatin", "Pravastatin", "Rosuvastatin",
    "Amlodipine", "Diltiazem", "Verapamil", "Nifedipine",
    "Metoprolol", "Atenolol", "Propranolol", "Carvedilol", "Bisoprolol",
    "Sertraline", "Fluoxetine", "Paroxetine", "Citalopram", "Escitalopram",
    "Alprazolam", "Lorazepam", "Diazepam", "Clonazepam",
    "Gabapentin", "Pregabalin", "Lamotrigine", "Levetiracetam",
    "Amoxicillin", "Azithromycin", "Clarithromycin", "Doxycycline",
    "Ciprofloxacin", "Levofloxacin", "Moxifloxacin",
    "Cephalexin", "Cefuroxime", "Ceftriaxone", "Cefdinir",
    "Prednisone", "Prednisolone", "Dexamethasone", "Hydrocortisone",
    "Levothyroxine", "Liothyronine",
    "Montelukast", "Fluticasone", "Budesonide", "Albuterol",
    "Warfarin", "Clopidogrel", "Rivaroxaban", "Apixaban",
    "Sildenafil", "Tadalafil",
    "Ondansetron", "Metoclopramide", "Promethazine",
    "Tramadol", "Codeine", "Hydrocodone",
    "Melatonin", "Vitamin D", "Folic acid", "Vitamin B12",
    "Finasteride", "Tamsulosin", "Dutasteride",
    "Acarbose", "Repaglinide", "Nateglinide",
    "Ezetimibe", "Fenofibrate", "Gemfibrozil",
    "Clonidine", "Methyldopa", "Hydralazine", "Minoxidil",
    "Amitriptyline", "Nortriptyline", "Duloxetine", "Venlafaxine",
    "Quetiapine", "Olanzapine", "Risperidone", "Aripiprazole",
    "Zolpidem", "Eszopiclone", "Trazodone",
]


class DIRILFetcher:
    """
    Fetch and process the FDA DIRIL (Drug-Induced Renal Injury List) dataset.

    The DIRIL dataset contains 317 curated drugs with nephrotoxicity labels:
    - 171 DIRI-positive (nephrotoxic)
    - 146 DIRI-negative (non-nephrotoxic)

    Source: FDA Bioinformatics Tools
    https://www.fda.gov/science-research/bioinformatics-tools/drug-induced-renal-injury-list-diril-dataset
    """

    DIRIL_URL = "https://www.fda.gov/media/178824/download?attachment"

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"]) / "nephrotox"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self):
        """
        Download and process the DIRIL dataset, or use curated fallback.

        Returns:
            pd.DataFrame: Processed dataset with SMILES and labels
        """
        logger.info("Fetching nephrotoxicity dataset...")

        # Try to download FDA DIRIL dataset
        excel_path = self.raw_dir / "diril_dataset.xlsx"
        csv_path = self.raw_dir / "nephrotox_curated_raw.csv"

        if excel_path.exists():
            logger.info(f"Using cached DIRIL dataset: {excel_path}")
            df = pd.read_excel(excel_path)
            df = self._process_diril(df)
        elif csv_path.exists():
            logger.info(f"Using cached curated dataset: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            # Try FDA download first
            try:
                logger.info("Attempting to download DIRIL dataset from FDA...")
                response = requests.get(
                    self.DIRIL_URL,
                    timeout=60,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; research)"}
                )
                response.raise_for_status()
                with open(excel_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded DIRIL dataset to {excel_path}")
                df = pd.read_excel(excel_path)
                df = self._process_diril(df)
            except Exception as e:
                logger.warning(f"FDA download failed: {e}")
                logger.info("Using curated nephrotoxicity dataset from literature...")
                df = self._create_curated_dataset()
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved curated data to {csv_path}")

        logger.info(f"Nephrotoxicity dataset: {len(df)} compounds")
        return df

    def _create_curated_dataset(self):
        """Create dataset from curated drug lists with ChEMBL SMILES lookup."""
        logger.info(f"Building curated dataset: {len(NEPHROTOXIC_DRUGS)} toxic, {len(NON_NEPHROTOXIC_DRUGS)} non-toxic drugs")

        records = []

        # Add nephrotoxic drugs
        for drug in NEPHROTOXIC_DRUGS:
            records.append({
                "drug_name": drug,
                "activity_label": 1,
                "activity_comment": "Toxic"
            })

        # Add non-nephrotoxic drugs
        for drug in NON_NEPHROTOXIC_DRUGS:
            records.append({
                "drug_name": drug,
                "activity_label": 0,
                "activity_comment": "Non-toxic"
            })

        df = pd.DataFrame(records)

        # Get SMILES from ChEMBL
        logger.info("Fetching SMILES from ChEMBL (this may take a few minutes)...")
        df = self._add_smiles_from_chembl(df)

        return df

    def _process_diril(self, df):
        """Process DIRIL dataset and add SMILES via ChEMBL lookup."""
        # Standardize column names
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        logger.info(f"Standardized columns: {list(df.columns)}")

        # Find the drug name and label columns
        name_col = None
        label_col = None

        for col in df.columns:
            if "drug" in col or "name" in col or "compound" in col:
                name_col = col
            if "diri" in col or "label" in col or "class" in col or "toxicity" in col:
                label_col = col

        if name_col is None:
            # Use first column as drug name
            name_col = df.columns[0]
        if label_col is None:
            # Use second column as label
            label_col = df.columns[1] if len(df.columns) > 1 else None

        logger.info(f"Using columns: name={name_col}, label={label_col}")

        # Create standardized dataframe
        result_df = pd.DataFrame()
        result_df["drug_name"] = df[name_col].astype(str).str.strip()

        if label_col:
            # Convert labels to binary (1 = toxic, 0 = non-toxic)
            labels = df[label_col].astype(str).str.lower()
            result_df["activity_label"] = labels.apply(
                lambda x: 1 if "positive" in x or "toxic" in x or x == "1" else 0
            )

        # Add activity comment for consistency
        result_df["activity_comment"] = result_df["activity_label"].apply(
            lambda x: "Toxic" if x == 1 else "Non-toxic"
        )

        # Get SMILES from ChEMBL
        logger.info("Fetching SMILES from ChEMBL for drug names...")
        result_df = self._add_smiles_from_chembl(result_df)

        # Log statistics
        n_toxic = result_df["activity_label"].sum()
        n_nontoxic = len(result_df) - n_toxic
        logger.info(f"DIRIL dataset: {n_toxic} toxic, {n_nontoxic} non-toxic")

        return result_df

    def _add_smiles_from_chembl(self, df):
        """Look up SMILES structures from ChEMBL using drug names."""
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, skipping SMILES lookup")
            df["canonical_smiles"] = None
            df["molecule_chembl_id"] = None
            return df

        molecule = new_client.molecule

        smiles_list = []
        chembl_ids = []

        for drug_name in df["drug_name"]:
            smiles = None
            chembl_id = None

            try:
                # Search by drug name (synonym)
                results = molecule.filter(
                    molecule_synonyms__molecule_synonym__iexact=drug_name
                ).only(["molecule_chembl_id", "molecule_structures"])

                if results:
                    mol = results[0]
                    chembl_id = mol.get("molecule_chembl_id")
                    structures = mol.get("molecule_structures")
                    if structures:
                        smiles = structures.get("canonical_smiles")

                # If not found, try pref_name
                if not smiles:
                    results = molecule.filter(
                        pref_name__iexact=drug_name
                    ).only(["molecule_chembl_id", "molecule_structures"])

                    if results:
                        mol = results[0]
                        chembl_id = mol.get("molecule_chembl_id")
                        structures = mol.get("molecule_structures")
                        if structures:
                            smiles = structures.get("canonical_smiles")

            except Exception as e:
                logger.debug(f"ChEMBL lookup failed for {drug_name}: {e}")

            smiles_list.append(smiles)
            chembl_ids.append(chembl_id)

        df["canonical_smiles"] = smiles_list
        df["molecule_chembl_id"] = chembl_ids

        # Filter to only compounds with SMILES
        n_before = len(df)
        df = df[df["canonical_smiles"].notna()].copy()
        n_after = len(df)

        logger.info(f"Found SMILES for {n_after}/{n_before} drugs ({100*n_after/n_before:.1f}%)")

        # Calculate molecular weight
        mol_weights = []
        for smiles in df["canonical_smiles"]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    from rdkit.Chem import Descriptors
                    mol_weights.append(Descriptors.MolWt(mol))
                else:
                    mol_weights.append(None)
            except:
                mol_weights.append(None)

        df["mol_wt"] = mol_weights

        # Generate InChIKey
        inchikeys = []
        for smiles in df["canonical_smiles"]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    inchikeys.append(Chem.MolToInchiKey(mol))
                else:
                    inchikeys.append(None)
            except:
                inchikeys.append(None)

        df["inchikey"] = inchikeys

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

        # Add task type
        df["task_type"] = "classification"

        return df
