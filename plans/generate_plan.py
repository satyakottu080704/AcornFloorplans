import sys
import os
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline import process_sketch


def _use_model():
    return os.getenv("PLAN_USE_MODEL", "true").strip().lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser(description="Generate Wates AI Draft Visio plan")
    parser.add_argument("project_number", help="Project number (e.g. N-12345)")
    parser.add_argument("--image", required=True, help="Path to surveyor sketch image")
    parser.add_argument("--format", default="vsdx", help="Output format")
    args = parser.parse_args()

    # Support both container path 'src/output/reports' (mapped in compose) and standard 'output/reports'
    output_dir = os.path.join(PROJECT_ROOT, "src", "output", "reports")
    if not os.path.exists(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, "output", "reports")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{args.project_number} AI Draft.vsdx")
    
    print(f"Generating plan for {args.project_number} from {args.image}...")
    print(f"Output path: {output_path}")

    # Run the image-based generation pipeline
    vsdx_path, plan = process_sketch(
        image_path=args.image,
        output_path=output_path,
        no_model=not _use_model()
    )
    print(f"Success! Generated: {vsdx_path}")

if __name__ == "__main__":
    main()
