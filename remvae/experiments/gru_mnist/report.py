import os
import sys
import json
import glob
import matplotlib.pyplot as plt
from datetime import timedelta

from generator.helpers.pdf import PDF


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_training_metrics(metrics, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Metrics Overview', fontsize=24, fontweight='bold')
    axes = axes.flatten()

    for idx, (key, values) in enumerate(metrics.items()):
        axes[idx].plot(values, color=f'C{idx}', marker='o', markersize=3, linewidth=2)
        axes[idx].set_title(key.replace('_', ' ').title(), fontsize=16)
        axes[idx].set_xlabel('Training Steps')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        axes[idx].set_facecolor('#f9f9f9')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

def format_training_time(seconds_str):
    try:
        seconds = int(float(seconds_str))
        return str(timedelta(seconds=seconds))
    except ValueError:
        return None

def flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def generate_report(args_path):
    args = load_json(args_path)

    BASE_DIR = os.path.dirname(__file__)
    METRICS_DIR = os.path.join(args["results_dir"], "metrics")
    IMAGES_DIR = os.path.join(BASE_DIR, 'images')
    OUTPUT_PDF = os.path.join(BASE_DIR, 'report.pdf')
    os.makedirs(IMAGES_DIR, exist_ok=True)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.add_section_title("0. Experiment Information")
    flat_args = flatten_dict(args)
    pdf.add_table(flat_args)

    pdf.add_page()
    pdf.add_section_title("1. Training")

    training_metrics_path = os.path.join(METRICS_DIR, 'training_metrics.json')
    metrics = load_json(training_metrics_path)

    plot_path = os.path.join(IMAGES_DIR, 'training_metrics.png')
    plot_training_metrics(metrics, plot_path)

    pdf.add_section_title("1.1 Training Metrics Chart")
    pdf.add_image(plot_path)

    time_path = os.path.join(METRICS_DIR, 'time')
    with open(time_path, 'r') as f:
        training_time = f.read()

    pdf.add_section_title("1.2 Training Time")
    pdf.add_text(f"Total training time: {training_time} seconds")

    formatted_time = format_training_time(training_time)
    if formatted_time:
        pdf.add_text(f"Formatted duration: {formatted_time} (hh:mm:ss)")
    else:
        pdf.add_text("Unable to parse training time.")

    pdf.add_page()
    pdf.add_section_title("2. Generated Images")
    sections = {
        'generated_image_from_text': "Images Generated from Text",
        'interpolation': "Interpolation"
    }

    subsection_count = 0
    for prefix, title in sections.items():
        image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, f"{prefix}*.png")))
        if image_files:
            if subsection_count == 2:
                pdf.add_page()
                subsection_count = 0
            pdf.add_section_title(title)
            for img_path in image_files:
                pdf.add_image(img_path, w=150)
            subsection_count += 1

    pdf.add_page()
    pdf.add_section_title("3. Evaluation Results")

    eval_files = glob.glob(os.path.join(METRICS_DIR, "eval*.json"))
    for eval_file in sorted(eval_files):
        eval_data = load_json(eval_file)
        eval_name = os.path.basename(eval_file).replace(".json", "")
        pdf.add_section_title(f"Results: {eval_name}")

        if isinstance(eval_data, dict):
            pdf.add_table(eval_data)
        elif isinstance(eval_data, list):
            for i, item in enumerate(eval_data):
                pdf.add_section_title(f"{eval_name} - Item {i+1}")
                if isinstance(item, dict):
                    pdf.add_table(item)
                else:
                    pdf.add_text(str(item))
        else:
            pdf.add_text(str(eval_data))

    pdf.output(OUTPUT_PDF)
    print(f"Report generated: {OUTPUT_PDF}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <path_to_args.json>")
        sys.exit(1)

    args_path = sys.argv[1]
    generate_report(args_path)