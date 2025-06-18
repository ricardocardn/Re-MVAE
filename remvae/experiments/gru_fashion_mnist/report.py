import os
import json
import glob
import matplotlib.pyplot as plt
from generator.helpers.pdf import PDF
from datetime import timedelta

BASE_DIR = os.path.dirname(__file__)
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)


pdf.add_page()
pdf.add_section_title("1. Training")

training_metrics_path = os.path.join(METRICS_DIR, 'training_metrics.json')
with open(training_metrics_path, 'r') as f:
    metrics = json.load(f)

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
plot_path = os.path.join(BASE_DIR, 'images/training_metrics.png')
plt.savefig(plot_path)
plt.close()

pdf.add_section_title("1.1 Training Metrics Chart")
pdf.add_image(plot_path)

time_path = os.path.join(METRICS_DIR, 'time')
with open(time_path, 'r') as f:
    training_time = f.read()

pdf.add_section_title("1.2 Training Time")
pdf.add_text(f"Total training time: {training_time} seconds")

try:
    seconds = int(float(training_time))
    readable_time = str(timedelta(seconds=seconds))
    pdf.add_text(f"Formatted duration: {readable_time} (hh:mm:ss)")
except ValueError:
    pdf.add_text("Unable to parse training time.")


pdf.add_page()
sections = {
    'generated_image_from_text': "Images Generated from Text",
    'interpolation': "Interpolation"
}
pdf.add_section_title("2. Generated Images")

subsection_count = 0
for prefix, section_title in sections.items():
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, f"{prefix}*.png")))
    if image_files:
        if subsection_count == 2:
            pdf.add_page()
            subsection_count = 0
        pdf.add_section_title(section_title)
        for img_path in image_files:
            pdf.add_image(img_path, w=150)
        subsection_count += 1


pdf.add_page()
pdf.add_section_title("3. Evaluation Results")
eval_files = glob.glob(os.path.join(METRICS_DIR, "eval*.json"))
for eval_file in sorted(eval_files):
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)
    eval_name = os.path.basename(eval_file).replace(".json", "")
    pdf.add_section_title(f"Results: {eval_name}")
    pdf.add_table(eval_data)


output_path = os.path.join(BASE_DIR, 'report.pdf')
pdf.output(output_path)
print(f"Report generated: {output_path}")