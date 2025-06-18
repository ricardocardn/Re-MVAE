import argparse
import json
import os
import shutil

from renders.training import generate_training_script
from renders.evaluation import generate_eval_script
from renders.visualization import generate_visualize_script


def load_config(args_path: str) -> dict:
    with open(args_path, 'r') as f:
        return json.load(f)


def get_output_dir(config: dict) -> str:
    output_dir = os.path.join('experiments', config['name'])
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def copy_file(file: str, output_dir: str):
    src_report = os.path.join('generator', 'tools', file)
    dst_report = os.path.join(output_dir, file)
    shutil.copy(src_report, dst_report)
    print(f"Copied {file} to {dst_report}")


def main():
    parser = argparse.ArgumentParser(description="Generate all scripts (train, eval, visualize and report) from args.json")
    parser.add_argument("args_path", type=str, help="Path to args.json")
    args = parser.parse_args()

    config = load_config(args.args_path)
    output_dir = get_output_dir(config)

    print("Generating training script...")
    generate_training_script(args.args_path)

    print("Generating evaluation script...")
    generate_eval_script(args.args_path)

    print("Generating visualization script...")
    generate_visualize_script(args.args_path)

    copy_file("report.py", output_dir)
    copy_file("execute.sh", output_dir)

    print("All scripts generated successfully.")


if __name__ == "__main__":
    main()