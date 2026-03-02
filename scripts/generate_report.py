#!/usr/bin/env python3
"""
Comprehensive PDF Toxicity Report Generator

Generates detailed toxicity assessment reports including:
- Multi-endpoint ML predictions
- Molecular docking results
- SHAP interpretation
- Structural alerts
- Risk assessment
- Design recommendations

Usage:
    python scripts/generate_report.py --smiles "CCO" --name "Ethanol" --output report.pdf
    python scripts/generate_report.py --input compounds.csv --output toxicity_report.pdf
    python scripts/generate_report.py --smiles "CC(=O)Nc1ccc(O)cc1" --include-docking
"""

import sys
import os
import argparse
import json
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import PDF libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics.charts.barcharts import HorizontalBarChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Installing reportlab...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics.charts.barcharts import HorizontalBarChart

# Try to import RDKit for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

import yaml
from loguru import logger


def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


class ToxicityReportGenerator:
    """Generate comprehensive toxicity assessment PDF reports."""

    # Color scheme
    COLORS = {
        "primary": colors.HexColor("#1A5276"),
        "secondary": colors.HexColor("#2C3E50"),
        "accent": colors.HexColor("#3498DB"),
        "success": colors.HexColor("#27AE60"),
        "warning": colors.HexColor("#F39C12"),
        "danger": colors.HexColor("#E74C3C"),
        "light": colors.HexColor("#ECF0F1"),
        "dark": colors.HexColor("#2C3E50"),
        "gray": colors.HexColor("#95A5A6"),
        "white": colors.white,
    }

    RISK_COLORS = {
        "high": colors.HexColor("#E74C3C"),
        "moderate": colors.HexColor("#F39C12"),
        "low": colors.HexColor("#27AE60"),
        "unknown": colors.HexColor("#95A5A6"),
    }

    ENDPOINT_INFO = {
        "herg": {
            "name": "hERG Cardiotoxicity",
            "task": "Regression (pIC50)",
            "target": "hERG potassium channel",
            "concern": "QT prolongation, cardiac arrhythmia"
        },
        "hepatotox": {
            "name": "Hepatotoxicity",
            "task": "Classification",
            "target": "Liver",
            "concern": "Drug-induced liver injury (DILI)"
        },
        "nephrotox": {
            "name": "Nephrotoxicity",
            "task": "Classification",
            "target": "Kidney",
            "concern": "Drug-induced kidney injury"
        },
        "ames": {
            "name": "Ames Mutagenicity",
            "task": "Classification",
            "target": "DNA",
            "concern": "Genotoxicity, carcinogenicity risk"
        },
        "skin_sens": {
            "name": "Skin Sensitization",
            "task": "Classification",
            "target": "Immune system",
            "concern": "Allergic contact dermatitis"
        },
        "cytotox": {
            "name": "Cytotoxicity",
            "task": "Classification",
            "target": "General cellular",
            "concern": "Cell death, tissue damage"
        },
        "reproductive_tox": {
            "name": "Reproductive Toxicity",
            "task": "Classification",
            "target": "Reproductive system",
            "concern": "Teratogenicity, fertility effects"
        },
    }

    def __init__(self, include_docking: bool = True, include_shap: bool = True):
        """Initialize report generator."""
        self.include_docking = include_docking
        self.include_shap = include_shap
        self.config = load_config()
        self.predictor = None
        self.docking_manager = None
        self._init_predictor()

    def _init_predictor(self):
        """Initialize ML predictor and optionally docking manager."""
        from src.api.predictor import ToxicityPredictor
        self.predictor = ToxicityPredictor(models_dir=str(PROJECT_ROOT / "models"))

        if self.include_docking:
            try:
                from src.docking import DockingManager
                docking_config = self.config.get("docking", {})
                self.docking_manager = DockingManager(
                    config=self.config,
                    structures_dir=str(PROJECT_ROOT / docking_config.get("structures_dir", "data/structures")),
                    engine=docking_config.get("engine", "vina")
                )
                if not self.docking_manager.is_available():
                    logger.warning("Docking engine not available")
                    self.docking_manager = None
            except Exception as e:
                logger.warning(f"Could not initialize docking: {e}")
                self.docking_manager = None

    def _get_styles(self):
        """Get custom paragraph styles."""
        styles = getSampleStyleSheet()

        # Title style
        styles.add(ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=28,
            spaceAfter=20,
            textColor=self.COLORS["primary"],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Subtitle
        styles.add(ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=30,
            textColor=self.COLORS["gray"],
            alignment=TA_CENTER
        ))

        # Section header
        styles.add(ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=self.COLORS["secondary"],
            borderWidth=0,
            borderPadding=0,
        ))

        # Subsection header
        styles.add(ParagraphStyle(
            'SubsectionHeader',
            parent=styles['Heading2'],
            fontSize=13,
            spaceBefore=15,
            spaceAfter=8,
            textColor=self.COLORS["primary"],
        ))

        # Compound name
        styles.add(ParagraphStyle(
            'CompoundName',
            parent=styles['Heading1'],
            fontSize=20,
            spaceBefore=0,
            spaceAfter=10,
            textColor=self.COLORS["primary"],
            backColor=self.COLORS["light"],
            borderWidth=2,
            borderColor=self.COLORS["accent"],
            borderPadding=12,
        ))

        # SMILES style
        styles.add(ParagraphStyle(
            'SMILES',
            parent=styles['Normal'],
            fontSize=9,
            fontName='Courier',
            backColor=colors.HexColor("#F8F9FA"),
            borderWidth=0.5,
            borderColor=self.COLORS["gray"],
            borderPadding=8,
            spaceAfter=10
        ))

        # Info box
        styles.add(ParagraphStyle(
            'InfoBox',
            parent=styles['Normal'],
            fontSize=10,
            backColor=colors.HexColor("#E8F4FD"),
            borderWidth=1,
            borderColor=self.COLORS["accent"],
            borderPadding=10,
            spaceAfter=10
        ))

        # Warning box
        styles.add(ParagraphStyle(
            'WarningBox',
            parent=styles['Normal'],
            fontSize=10,
            backColor=colors.HexColor("#FFF3CD"),
            borderWidth=1,
            borderColor=self.COLORS["warning"],
            borderPadding=10,
            spaceAfter=10
        ))

        # Danger box
        styles.add(ParagraphStyle(
            'DangerBox',
            parent=styles['Normal'],
            fontSize=10,
            backColor=colors.HexColor("#F8D7DA"),
            borderWidth=1,
            borderColor=self.COLORS["danger"],
            borderPadding=10,
            spaceAfter=10
        ))

        # Success box
        styles.add(ParagraphStyle(
            'SuccessBox',
            parent=styles['Normal'],
            fontSize=10,
            backColor=colors.HexColor("#D4EDDA"),
            borderWidth=1,
            borderColor=self.COLORS["success"],
            borderPadding=10,
            spaceAfter=10
        ))

        return styles

    def _generate_molecule_image(self, smiles: str, size=(300, 300)) -> Optional[io.BytesIO]:
        """Generate molecule image from SMILES."""
        if not RDKIT_AVAILABLE:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            img = Draw.MolToImage(mol, size=size)
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            return img_buffer
        except Exception as e:
            logger.error(f"Error generating molecule image: {e}")
            return None

    def _calculate_mol_properties(self, smiles: str) -> Dict[str, Any]:
        """Calculate basic molecular properties."""
        if not RDKIT_AVAILABLE:
            return {}

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            return {
                "Molecular Weight": f"{Descriptors.MolWt(mol):.2f}",
                "LogP": f"{Descriptors.MolLogP(mol):.2f}",
                "TPSA": f"{Descriptors.TPSA(mol):.2f}",
                "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
                "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
                "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "Heavy Atoms": mol.GetNumHeavyAtoms(),
            }
        except Exception as e:
            logger.error(f"Error calculating properties: {e}")
            return {}

    def _create_risk_gauge(self, risk_score: float, width: float = 200, height: float = 60) -> Drawing:
        """Create a risk gauge visualization."""
        d = Drawing(width, height)

        # Background bar
        d.add(Rect(0, 20, width, 20, fillColor=colors.HexColor("#E0E0E0"), strokeColor=None))

        # Colored segments
        segment_width = width / 3
        d.add(Rect(0, 20, segment_width, 20, fillColor=self.RISK_COLORS["low"], strokeColor=None))
        d.add(Rect(segment_width, 20, segment_width, 20, fillColor=self.RISK_COLORS["moderate"], strokeColor=None))
        d.add(Rect(segment_width * 2, 20, segment_width, 20, fillColor=self.RISK_COLORS["high"], strokeColor=None))

        # Indicator position
        indicator_x = min(width - 5, max(5, risk_score * width))
        d.add(Rect(indicator_x - 3, 15, 6, 30, fillColor=colors.black, strokeColor=None))

        # Labels
        d.add(String(segment_width / 2, 5, "Low", fontSize=8, textAnchor='middle'))
        d.add(String(segment_width * 1.5, 5, "Moderate", fontSize=8, textAnchor='middle'))
        d.add(String(segment_width * 2.5, 5, "High", fontSize=8, textAnchor='middle'))

        return d

    def _create_title_page(self, compounds: Dict[str, str], styles) -> List:
        """Create title page elements."""
        elements = []

        elements.append(Spacer(1, 1.5 * inch))
        elements.append(Paragraph("Multi-Endpoint Toxicity", styles['CustomTitle']))
        elements.append(Paragraph("Prediction Report", styles['CustomTitle']))
        elements.append(Spacer(1, 0.3 * inch))

        # Subtitle with date
        date_str = datetime.now().strftime('%B %d, %Y at %H:%M')
        elements.append(Paragraph(f"Generated: {date_str}", styles['Subtitle']))
        elements.append(Spacer(1, 0.5 * inch))

        # Report summary box
        summary_data = [
            ["Report Summary", ""],
            ["Compounds Analyzed", str(len(compounds))],
            ["Toxicity Endpoints", str(len(self.ENDPOINT_INFO))],
            ["Docking Included", "Yes" if self.docking_manager else "No"],
            ["SHAP Analysis", "Yes" if self.include_shap else "No"],
        ]

        summary_table = Table(summary_data, colWidths=[2.5 * inch, 2 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS["primary"]),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('SPAN', (0, 0), (-1, 0)),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BACKGROUND', (0, 1), (0, -1), self.COLORS["light"]),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, self.COLORS["accent"]),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.5 * inch))

        # Compounds list
        elements.append(Paragraph("Compounds in This Report:", styles['SubsectionHeader']))
        for name in compounds.keys():
            elements.append(Paragraph(f"• {name}", styles['Normal']))

        elements.append(PageBreak())
        return elements

    def _create_compound_section(
        self,
        compound_name: str,
        smiles: str,
        ml_results: Dict,
        docking_results: Optional[Dict],
        styles
    ) -> List:
        """Create PDF elements for a single compound."""
        elements = []

        # Compound header
        elements.append(Paragraph(f"{compound_name}", styles['CompoundName']))
        elements.append(Paragraph(f"<b>SMILES:</b> {smiles}", styles['SMILES']))

        # Two-column layout: molecule image and properties
        mol_img = self._generate_molecule_image(smiles)
        mol_props = self._calculate_mol_properties(smiles)

        if mol_img and mol_props:
            # Properties table
            prop_data = [[k, str(v)] for k, v in mol_props.items()]
            prop_table = Table(prop_data, colWidths=[1.5 * inch, 1 * inch])
            prop_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), self.COLORS["light"]),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS["gray"]),
            ]))

            # Combine image and properties
            main_table = Table([
                [Image(mol_img, width=2.5 * inch, height=2.5 * inch), prop_table]
            ], colWidths=[3 * inch, 3 * inch])
            main_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ]))
            elements.append(main_table)
        elif mol_img:
            elements.append(Image(mol_img, width=2.5 * inch, height=2.5 * inch))

        elements.append(Spacer(1, 15))

        # Check if prediction was successful
        if not ml_results.get("success", False):
            elements.append(Paragraph(
                f"<b>Error:</b> {ml_results.get('error', 'Unknown error')}",
                styles['DangerBox']
            ))
            return elements

        # === INTEGRATED RISK ASSESSMENT ===
        assessment = ml_results.get("integrated_assessment", {})
        if assessment:
            elements.append(Paragraph("Integrated Risk Assessment", styles['SectionHeader']))
            elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["accent"]))

            risk_level = assessment.get("overall_risk_level", "unknown")
            risk_score = assessment.get("overall_risk_score", 0)
            critical_endpoint = assessment.get("critical_endpoint", "N/A")

            # Risk summary
            risk_color = self.RISK_COLORS.get(risk_level, self.RISK_COLORS["unknown"])

            risk_data = [
                ["OVERALL RISK", risk_level.upper()],
                ["Risk Score", f"{risk_score:.3f}"],
                ["Critical Endpoint", critical_endpoint],
            ]

            risk_table = Table(risk_data, colWidths=[2.5 * inch, 3 * inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), self.COLORS["secondary"]),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
                ('BACKGROUND', (1, 0), (1, 0), risk_color),
                ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 1), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ]))
            elements.append(risk_table)
            elements.append(Spacer(1, 10))

            # Risk gauge
            elements.append(self._create_risk_gauge(risk_score))
            elements.append(Spacer(1, 10))

            # Recommendation
            if assessment.get("recommendation"):
                box_style = 'DangerBox' if risk_level == 'high' else 'WarningBox' if risk_level == 'moderate' else 'SuccessBox'
                elements.append(Paragraph(
                    f"<b>Recommendation:</b> {assessment['recommendation']}",
                    styles[box_style]
                ))

        # === ENDPOINT PREDICTIONS ===
        elements.append(Paragraph("Toxicity Endpoint Predictions", styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["accent"]))

        predictions = ml_results.get("predictions", {})
        pred_data = [["Endpoint", "Result", "Probability", "Risk", "In AD"]]

        for endpoint, pred in predictions.items():
            ep_info = self.ENDPOINT_INFO.get(endpoint, {"name": endpoint})

            if "error" in pred:
                pred_data.append([ep_info["name"], "ERROR", "-", "-", "-"])
            else:
                task_type = pred.get("task_type", "classification")
                if task_type == "classification":
                    result = pred.get("label", str(pred.get("prediction", "N/A")))
                    prob = f"{pred.get('probability', 0):.1%}"
                else:
                    result = f"{pred.get('prediction', 0):.2f}"
                    prob = pred.get("unit", "pIC50")

                ad = pred.get("applicability_domain", {})
                in_ad = "✓" if ad.get("in_domain", False) else "✗"

                # Determine risk level
                prob_val = pred.get("probability", 0.5)
                if prob_val >= 0.7:
                    risk = "High"
                elif prob_val >= 0.3:
                    risk = "Moderate"
                else:
                    risk = "Low"

                pred_data.append([ep_info["name"], result, prob, risk, in_ad])

        pred_table = Table(pred_data, colWidths=[2 * inch, 1.5 * inch, 1 * inch, 1 * inch, 0.7 * inch])
        table_styles = [
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS["secondary"]),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS["gray"]),
        ]

        # Color-code risk column
        for i, row in enumerate(pred_data[1:], 1):
            if row[3] == "High":
                table_styles.append(('BACKGROUND', (3, i), (3, i), colors.HexColor("#FFCDD2")))
            elif row[3] == "Moderate":
                table_styles.append(('BACKGROUND', (3, i), (3, i), colors.HexColor("#FFE0B2")))
            elif row[3] == "Low":
                table_styles.append(('BACKGROUND', (3, i), (3, i), colors.HexColor("#C8E6C9")))

        pred_table.setStyle(TableStyle(table_styles))
        elements.append(pred_table)
        elements.append(Spacer(1, 15))

        # === DOCKING RESULTS ===
        if docking_results and any(r.get("success") or r.get("docking", {}).get("success") for r in docking_results.values()):
            elements.append(Paragraph("Molecular Docking Results", styles['SectionHeader']))
            elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["accent"]))

            dock_data = [["Target", "Affinity (kcal/mol)", "Enhanced Score", "Risk Level"]]

            for target, result in docking_results.items():
                # Handle both old and new result formats
                if isinstance(result, dict) and "docking" in result:
                    docking = result.get("docking", {})
                    enhanced = result.get("enhanced_score", {})
                    success = docking.get("success", False)
                    affinity = docking.get("affinity", 0)
                    enhanced_score = enhanced.get("score", 0)
                    risk = enhanced.get("risk_level", "unknown").capitalize()
                else:
                    success = result.get("success", False)
                    affinity = result.get("affinity", 0)
                    enhanced_score = result.get("normalized_score", 0)
                    if affinity is not None and affinity < -8:
                        risk = "High"
                    elif affinity is not None and affinity < -6:
                        risk = "Moderate"
                    else:
                        risk = "Low"

                if success:
                    dock_data.append([
                        target.upper(),
                        f"{affinity:.1f}" if affinity else "N/A",
                        f"{enhanced_score:.3f}" if enhanced_score else "N/A",
                        risk
                    ])

            if len(dock_data) > 1:
                dock_table = Table(dock_data, colWidths=[1.8 * inch, 1.5 * inch, 1.2 * inch, 1 * inch])
                dock_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.COLORS["primary"]),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS["gray"]),
                ]))
                elements.append(dock_table)
                elements.append(Spacer(1, 8))

                elements.append(Paragraph(
                    "<i>Interpretation: More negative affinity = stronger binding = higher toxicity risk. "
                    "Enhanced score incorporates 3D shape and pharmacophore compatibility.</i>",
                    styles['Normal']
                ))

        elements.append(Spacer(1, 15))

        # === 3D DESCRIPTORS ===
        descriptors_3d = docking_results.get("descriptors_3d", {}) if isinstance(docking_results, dict) else {}
        if descriptors_3d and descriptors_3d.get("success"):
            elements.append(Paragraph("3D Molecular Descriptors", styles['SectionHeader']))
            elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["accent"]))

            # Shape descriptors
            shape = descriptors_3d.get("shape_descriptors", {})
            if shape:
                elements.append(Paragraph("<b>Shape Properties</b>", styles['SubsectionHeader']))
                shape_data = [
                    ["Property", "Value", "Property", "Value"],
                    ["Asphericity", f"{shape.get('asphericity', 0):.3f}",
                     "Eccentricity", f"{shape.get('eccentricity', 0):.3f}"],
                    ["NPR1", f"{shape.get('npr1', 0):.3f}",
                     "NPR2", f"{shape.get('npr2', 0):.3f}"],
                    ["Radius of Gyration", f"{shape.get('radius_of_gyration', 0):.2f} Å",
                     "Spherocity Index", f"{shape.get('spherocity_index', 0):.3f}"],
                ]
                shape_table = Table(shape_data, colWidths=[1.5 * inch, 1.2 * inch, 1.5 * inch, 1.2 * inch])
                shape_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.COLORS["light"]),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                    ('ALIGN', (3, 0), (3, -1), 'CENTER'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS["gray"]),
                ]))
                elements.append(shape_table)
                elements.append(Spacer(1, 10))

            # Volume and surface
            volume = descriptors_3d.get("volume_descriptors", {})
            surface = descriptors_3d.get("surface_descriptors", {})
            if volume or surface:
                vol_surf_data = [
                    ["Molecular Volume", f"{volume.get('molecular_volume', 0):.1f} Å³",
                     "TPSA", f"{surface.get('tpsa_3d', 0):.1f} Å²"],
                    ["SASA", f"{surface.get('sasa', 0):.1f} Å²", "", ""],
                ]
                vol_table = Table(vol_surf_data, colWidths=[1.5 * inch, 1.2 * inch, 1.5 * inch, 1.2 * inch])
                vol_table.setStyle(TableStyle([
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                    ('ALIGN', (3, 0), (3, -1), 'CENTER'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS["gray"]),
                ]))
                elements.append(vol_table)
                elements.append(Spacer(1, 10))

        # === PHARMACOPHORE FEATURES ===
        pharm = descriptors_3d.get("pharmacophore_counts", {}) if descriptors_3d else {}
        if pharm:
            elements.append(Paragraph("Pharmacophore Profile", styles['SectionHeader']))
            elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["accent"]))

            pharm_data = [
                ["Feature Type", "Count"],
                ["H-Bond Acceptors", str(pharm.get("h_bond_acceptors", 0))],
                ["H-Bond Donors", str(pharm.get("h_bond_donors", 0))],
                ["Aromatic Rings", str(pharm.get("aromatic_rings", 0))],
                ["Hydrophobic Centers", str(pharm.get("hydrophobic_centers", 0))],
                ["Positive Ionizable", str(pharm.get("positive_ionizable", 0))],
                ["Negative Ionizable", str(pharm.get("negative_ionizable", 0))],
            ]

            pharm_table = Table(pharm_data, colWidths=[2.5 * inch, 1.5 * inch])
            pharm_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.COLORS["secondary"]),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS["gray"]),
            ]))
            elements.append(pharm_table)
            elements.append(Spacer(1, 8))

            # Pharmacophore interpretation
            total_features = sum(pharm.values())
            interpretation = []
            if pharm.get("aromatic_rings", 0) >= 2:
                interpretation.append("Multiple aromatic rings may increase hERG binding risk")
            if pharm.get("h_bond_acceptors", 0) >= 5:
                interpretation.append("High HBA count - good for solubility, may affect permeability")
            if pharm.get("positive_ionizable", 0) >= 1:
                interpretation.append("Positive ionizable groups increase CYP2D6 binding likelihood")

            if interpretation:
                elements.append(Paragraph("<b>Pharmacophore Notes:</b>", styles['Normal']))
                for note in interpretation:
                    elements.append(Paragraph(f"• {note}", styles['Normal']))

            elements.append(Spacer(1, 10))

        # === BINDING COMPATIBILITY ===
        # Check for binding compatibility data in docking results
        compatibility_shown = False
        if isinstance(docking_results, dict):
            for target, result in docking_results.items():
                if target == "descriptors_3d":
                    continue
                if isinstance(result, dict) and "binding_compatibility" in result:
                    if not compatibility_shown:
                        elements.append(Paragraph("Target Binding Compatibility", styles['SectionHeader']))
                        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["accent"]))
                        compatibility_shown = True

                    compat = result["binding_compatibility"]
                    compat_text = f"<b>{target.upper()}:</b> "
                    compat_text += f"Shape: {compat.get('shape_compatibility', 'N/A')}, "
                    compat_text += f"Size: {compat.get('size_compatibility', 'N/A')}, "
                    compat_text += f"Pharmacophore: {compat.get('pharmacophore_match', 'N/A')}"

                    risk = compat.get("binding_risk", "unknown")
                    if risk == "high":
                        elements.append(Paragraph(compat_text, styles['DangerBox']))
                    elif risk == "moderate":
                        elements.append(Paragraph(compat_text, styles['WarningBox']))
                    else:
                        elements.append(Paragraph(compat_text, styles['SuccessBox']))

                    # Show recommendations
                    for rec in compat.get("recommendations", []):
                        elements.append(Paragraph(f"  → {rec}", styles['Normal']))

        if compatibility_shown:
            elements.append(Spacer(1, 15))

        # === STRUCTURAL ALERTS ===
        structural_alerts = ml_results.get("structural_alerts", [])
        if structural_alerts:
            elements.append(Paragraph("Structural Alerts", styles['SectionHeader']))
            elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["danger"]))

            for alert in structural_alerts:
                if isinstance(alert, dict):
                    alert_text = f"<b>{alert.get('name', 'Alert')}:</b> {alert.get('description', '')}"
                    if alert.get('severity'):
                        alert_text += f" (Severity: {alert['severity']})"
                else:
                    alert_text = str(alert)
                elements.append(Paragraph(f"⚠️ {alert_text}", styles['DangerBox']))

        # === DESIGN RECOMMENDATIONS ===
        recommendations = ml_results.get("design_recommendations", [])
        if recommendations:
            elements.append(Paragraph("Design Recommendations", styles['SectionHeader']))
            elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["success"]))

            for rec in recommendations:
                if isinstance(rec, dict):
                    rec_text = f"<b>{rec.get('type', 'Tip')}:</b> {rec.get('suggestion', str(rec))}"
                else:
                    rec_text = str(rec)
                elements.append(Paragraph(f"💡 {rec_text}", styles['InfoBox']))

        return elements

    def _create_methodology_section(self, styles) -> List:
        """Create methodology section."""
        elements = []

        elements.append(Paragraph("Methodology", styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["accent"]))

        method_sections = [
            ("<b>Feature Engineering</b>", [
                "RDKit Molecular Descriptors (25 physicochemical properties)",
                "Morgan Fingerprints (2048 bits, radius=2)",
                "MACCS Structural Keys (167 bits)",
            ]),
            ("<b>Machine Learning Models</b>", [
                "Random Forest, XGBoost, LightGBM, SVM classifiers",
                "Hyperparameter optimization via Optuna",
                "5-fold stratified cross-validation",
            ]),
            ("<b>Molecular Docking</b>", [
                "AutoDock Vina for binding affinity prediction",
                "Protein targets: hERG, CYP3A4, CYP2D6, CYP2C9, ER-α, AR",
                "Ensemble scoring combining ML + docking",
            ]),
            ("<b>3D Descriptors &amp; Pharmacophores</b>", [
                "ETKDG conformer generation with MMFF optimization",
                "Shape descriptors: asphericity, eccentricity, PMI ratios",
                "Surface properties: TPSA, SASA, molecular volume",
                "Pharmacophore features: HBA, HBD, aromatic, hydrophobic, ionizable",
                "Gobbi 2D pharmacophore fingerprints for similarity analysis",
            ]),
            ("<b>Interpretability</b>", [
                "SHAP values for feature importance",
                "Structural alert detection (toxicophores)",
                "Applicability domain assessment",
                "Target binding compatibility analysis",
            ]),
        ]

        for title, items in method_sections:
            elements.append(Paragraph(title, styles['SubsectionHeader']))
            for item in items:
                elements.append(Paragraph(f"• {item}", styles['Normal']))
            elements.append(Spacer(1, 8))

        return elements

    def _create_disclaimer(self, styles) -> List:
        """Create disclaimer section."""
        elements = []

        disclaimer_text = """
        <b>Disclaimer:</b> This report is generated by computational models for research and screening
        purposes only. The predictions are based on machine learning models trained on publicly available
        datasets and should not be used as the sole basis for regulatory decisions or clinical applications.
        Experimental validation is required to confirm any predicted toxicity. The accuracy of predictions
        depends on the chemical similarity of query compounds to the training data (applicability domain).
        Always consult domain experts and perform appropriate experimental validation before making
        decisions based on these predictions.
        """

        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=self.COLORS["gray"],
            backColor=self.COLORS["light"],
            borderWidth=1,
            borderColor=self.COLORS["gray"],
            borderPadding=12,
            alignment=TA_JUSTIFY,
            leading=12
        )

        elements.append(Spacer(1, 20))
        elements.append(Paragraph(disclaimer_text, disclaimer_style))

        return elements

    def generate_report(
        self,
        compounds: Dict[str, str],
        output_path: str,
        title: str = None
    ) -> str:
        """
        Generate comprehensive PDF toxicity report.

        Args:
            compounds: Dictionary of {name: smiles}
            output_path: Output PDF file path
            title: Optional custom title

        Returns:
            Path to generated PDF
        """
        logger.info(f"Generating toxicity report for {len(compounds)} compounds")

        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.6 * inch,
            leftMargin=0.6 * inch,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch
        )

        styles = self._get_styles()
        elements = []

        # Title page
        elements.extend(self._create_title_page(compounds, styles))

        # Process each compound
        for compound_name, smiles in compounds.items():
            logger.info(f"Processing {compound_name}...")

            # Get ML predictions
            try:
                if self.include_shap:
                    ml_results = self.predictor.predict_multi_with_interpretation(smiles, top_k=10)
                else:
                    ml_results = self.predictor.predict_multi_endpoint(smiles)
            except Exception as e:
                logger.error(f"ML prediction failed for {compound_name}: {e}")
                ml_results = {"success": False, "error": str(e)}

            # Get docking results with 3D descriptors
            docking_results = {}
            if self.docking_manager and ml_results.get("success"):
                for endpoint in ["herg", "hepatotox", "reproductive_tox"]:
                    try:
                        # Use enhanced docking with 3D descriptors
                        enhanced_results = self.docking_manager.dock_for_endpoint_enhanced(smiles, endpoint)

                        # Store 3D descriptors (only once)
                        if "descriptors_3d" in enhanced_results and "descriptors_3d" not in docking_results:
                            docking_results["descriptors_3d"] = enhanced_results["descriptors_3d"]

                        # Store target results
                        for target, result in enhanced_results.get("targets", {}).items():
                            docking_results[target] = result
                    except Exception as e:
                        logger.warning(f"Enhanced docking failed for {endpoint}: {e}")
                        # Fallback to basic docking
                        try:
                            target_results = self.docking_manager.dock_for_endpoint(smiles, endpoint)
                            for target, result in target_results.items():
                                docking_results[target] = result.to_dict() if hasattr(result, 'to_dict') else result
                        except:
                            pass

            # Create compound section
            elements.extend(self._create_compound_section(
                compound_name, smiles, ml_results, docking_results, styles
            ))
            elements.append(PageBreak())

        # Methodology
        elements.extend(self._create_methodology_section(styles))

        # Disclaimer
        elements.extend(self._create_disclaimer(styles))

        # Build PDF
        logger.info(f"Building PDF: {output_path}")
        doc.build(elements)
        logger.info(f"Report generated successfully: {output_path}")

        return output_path


def load_compounds_from_file(filepath: str) -> Dict[str, str]:
    """Load compounds from CSV file."""
    import pandas as pd

    df = pd.read_csv(filepath)

    # Find columns
    smiles_col = None
    for col in ['smiles', 'SMILES', 'Smiles', 'canonical_smiles']:
        if col in df.columns:
            smiles_col = col
            break

    name_col = None
    for col in ['name', 'Name', 'id', 'ID', 'compound_id']:
        if col in df.columns:
            name_col = col
            break

    if not smiles_col:
        raise ValueError("No SMILES column found")

    compounds = {}
    for idx, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()
        if not smiles or smiles == 'nan':
            continue
        name = str(row[name_col]) if name_col else f"Compound_{idx + 1}"
        compounds[name] = smiles

    return compounds


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive PDF toxicity report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_report.py --smiles "CCO" --name "Ethanol"
  python scripts/generate_report.py --input compounds.csv --output report.pdf
  python scripts/generate_report.py --smiles "CC(=O)Nc1ccc(O)cc1" --name "Acetaminophen" --include-docking
        """
    )

    parser.add_argument("--smiles", "-s", help="SMILES string (comma-separated for multiple)")
    parser.add_argument("--name", "-n", help="Compound name (comma-separated for multiple)")
    parser.add_argument("--input", "-i", help="Input CSV file with compounds")
    parser.add_argument("--output", "-o", default="reports/toxicity_report.pdf", help="Output PDF path")
    parser.add_argument("--include-docking", action="store_true", help="Include molecular docking")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP analysis")

    args = parser.parse_args()

    # Load compounds
    compounds = {}

    if args.input:
        compounds = load_compounds_from_file(args.input)
    elif args.smiles:
        smiles_list = [s.strip() for s in args.smiles.split(",")]
        if args.name:
            names = [n.strip() for n in args.name.split(",")]
        else:
            names = [f"Compound_{i + 1}" for i in range(len(smiles_list))]

        for name, smiles in zip(names, smiles_list):
            compounds[name] = smiles
    else:
        # Default test compounds
        compounds = {
            "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        }

    if not compounds:
        print("No compounds to process")
        sys.exit(1)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate report
    generator = ToxicityReportGenerator(
        include_docking=args.include_docking,
        include_shap=not args.no_shap
    )

    report_path = generator.generate_report(compounds, str(output_path))
    print(f"\nReport generated: {report_path}")


if __name__ == "__main__":
    main()
