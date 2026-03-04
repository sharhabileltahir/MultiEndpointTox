"""Test script for molecular docking setup."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("MultiEndpointTox - Docking Setup Test")
    print("=" * 60)
    print()

    # Check dependencies
    try:
        from src.docking.utils import check_docking_dependencies, get_installation_instructions

        status = check_docking_dependencies()

        print("Dependency Status:")
        print(f"  RDKit:        {'✓' if status['rdkit'] else '✗'}")
        print(f"  Meeko:        {'✓' if status['meeko'] else '✗'} (optional)")
        print(f"  Open Babel:   {'✓' if status['openbabel'] else '✗'} (optional)")
        print(f"  Vina Python:  {'✓' if status['vina_python'] else '✗'}")
        print(f"  Vina CLI:     {'✓' if status['vina_cli'] else '✗'}")
        if status['vina_path']:
            print(f"  Vina Path:    {status['vina_path']}")
        print()
        print(f"Ready for docking: {'✓ YES' if status['ready'] else '✗ NO'}")
        print()

        if not status['ready']:
            print(get_installation_instructions())
            return 1

        # Test docking if ready
        print("Testing docking functionality...")
        print()

        from src.docking import DockingManager
        import yaml

        # Load config
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize docking manager
        dm = DockingManager(
            config=config,
            structures_dir=str(PROJECT_ROOT / "data" / "structures"),
            engine=config.get("docking", {}).get("engine", "vina")
        )

        print(f"Docking manager initialized: {dm.engine_name}")
        print(f"Engine available: {dm.is_available()}")
        print()

        if dm.is_available():
            # Test with a simple molecule (aspirin)
            test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
            print(f"Test docking: {test_smiles}")
            print("Target: herg")
            print()

            result = dm.dock(test_smiles, "herg", exhaustiveness=4, n_poses=3)

            if result.success:
                print(f"✓ Docking successful!")
                print(f"  Affinity: {result.affinity:.2f} kcal/mol")
                print(f"  Poses: {result.num_poses}")
                print(f"  Time: {result.execution_time:.1f}s")
            else:
                print(f"✗ Docking failed: {result.error}")

        print()
        print("=" * 60)
        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        print()
        print("Make sure the docking module is properly installed.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
