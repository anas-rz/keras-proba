"""Generates the documentation pages of the example notebooks.

Scans `examples/<task>/<group>/*.ipynb`, where `<group>` is `training` or `postprocessing`,
and writes (build output only: `docs/examples/` is not committed):

1. `docs/examples/index.md`: the tasks, and every example.
2. `docs/examples/<task>.md`: a gallery of the task's examples, grouped into training and
   post-processing, with links to the tutorial, Colab and GitHub.
3. `docs/examples/<notebook>.md`: one page per notebook, with "View in Colab" and "GitHub source"
   buttons, followed by the notebook's markdown, code and saved outputs (text and plots).

Titles, descriptions and the techniques used are read from the notebooks themselves, so the
pages always follow them.
"""

import glob
import json
import os
import re

REPO_URL = "https://github.com/anas-rz/keras-proba/blob/main"
COLAB_URL = "https://colab.research.google.com/github/anas-rz/keras-proba/blob/main"
EXAMPLES_DIR = "examples"
OUTPUT_DIR = os.path.join("docs", "examples")

TASKS = {
    "forecasting": {
        "title": "Forecasting",
        "icon": ":material-weather-partly-cloudy:",
        "dataset": "Jena climate",
        "description": "Forecast the temperature 24 hours ahead: at a weather station from its last 48 hours of "
        "measurements, or everywhere on Earth with a pre-trained global weather model.",
    },
    "regression": {
        "title": "Regression",
        "icon": ":material-home-city-outline:",
        "dataset": "California housing",
        "description": "Predict house values in California districts from census data.",
    },
    "classification": {
        "title": "Classification",
        "icon": ":material-tshirt-crew-outline:",
        "dataset": "Fashion-MNIST, MNIST",
        "description": "Classify clothing images, and recognize inputs unlike the training data (handwritten digits).",
    },
}
# Notebooks whose dataset differs from their task's.
DATASETS = {
    "pretrained_fourcastnet_conformal": "ERA5 (WeatherBench 2)",
}
GROUPS = {
    "training": ("Training", "Make the model itself uncertainty-aware."),
    "postprocessing": ("Post-processing", "Add calibrated uncertainty to an already trained model, without retraining it."),
}


def read_notebook(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cells"]


def source(cell):
    return "".join(cell.get("source", [])).strip()


def notebook_meta(path):
    rel = os.path.relpath(path, EXAMPLES_DIR)
    task, group = rel.split(os.sep)[:2]
    cells = read_notebook(path)
    markdown = [source(c) for c in cells if c["cell_type"] == "markdown"]
    code = "\n".join(source(c) for c in cells if c["cell_type"] == "code")

    title, _, intro = markdown[0].partition("\n")
    intro = " ".join(intro.strip().split("\n\n")[0].split())
    intro = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", intro)  # Markdown links -> text.
    intro = re.sub(r"\*\*([^*]+)\*\*", r"\1", intro)
    description = re.split(r"(?<=[.!?])\s", intro, maxsplit=1)[0]

    used = re.findall(r"keras_proba\.(?:layers|postprocessing|callbacks)\.([A-Z]\w+)", code)
    techniques = list(dict.fromkeys(used))  # Unique, in order of first use.

    stem = os.path.splitext(os.path.basename(path))[0]
    path = path.replace(os.sep, "/")
    return {
        "title": title.lstrip("# ").strip(),
        "description": description,
        "techniques": techniques,
        "task": task,
        "group": group,
        "dataset": DATASETS.get(stem, TASKS[task]["dataset"]) if task in TASKS else None,
        "stem": stem,
        "doc_file": f"{stem}.md",
        "github_url": f"{REPO_URL}/{path}",
        "colab_url": f"{COLAB_URL}/{path}",
    }


def render_outputs(cell):
    """Saved outputs of a code cell, as text blocks and inline images."""
    lines = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        data = output.get("data", {})
        if kind == "stream" and output.get("name") == "stdout":
            text = "".join(output.get("text", [])).strip()
            if text:
                lines += ["```text", text, "```", ""]
        elif kind in ("display_data", "execute_result"):
            if "image/png" in data:
                image = "".join(data["image/png"]).replace("\n", "")
                lines += [f'<img alt="Output plot" src="data:image/png;base64,{image}" />', ""]
            elif "text/plain" in data and kind == "execute_result":
                lines += ["```text", "".join(data["text/plain"]).strip(), "```", ""]
    return lines


def example_page(path, meta):
    task = TASKS[meta["task"]]
    lines = [
        f"# {meta['title']}",
        "",
        f"**Backend:** TensorFlow, JAX or PyTorch<br>",
        f"**Dataset:** {meta['dataset']}<br>",
        f"**Techniques:** {', '.join(f'`{t}`' for t in meta['techniques'])}",
        "",
        f'[:simple-googlecolab: **View in Colab**]({meta["colab_url"]}){{ .md-button .md-button--primary }} &nbsp; '
        f'[:octicons-mark-github-16: **GitHub source**]({meta["github_url"]}){{ .md-button }}',
        "",
        "---",
        "",
    ]
    for index, cell in enumerate(read_notebook(path)):
        text = source(cell)
        if not text:
            continue
        if cell["cell_type"] == "markdown":
            if index == 0:
                # The title is already the page heading: keep the introduction.
                text = text.partition("\n")[2].strip()
            lines += [text, ""]
        elif cell["cell_type"] == "code":
            lines += ["```python", text, "```", ""]
            lines += render_outputs(cell)
    return "\n".join(lines)


def card(meta):
    techniques = " · ".join(f"`{t}`" for t in meta["techniques"][:3])
    return [
        f"-   __{meta['title']}__",
        "",
        f"    {meta['description']}",
        "",
        f"    {techniques}",
        "",
        f"    [:octicons-arrow-right-24: Read tutorial]({meta['doc_file']}){{ .md-button .md-button--primary }} &nbsp; "
        f'[:simple-googlecolab:]({meta["colab_url"]}){{ .md-button title="View in Colab" }} &nbsp; '
        f'[:octicons-mark-github-16:]({meta["github_url"]}){{ .md-button title="GitHub source" }}',
        "",
    ]


def task_page(task, examples):
    meta = TASKS[task]
    lines = [
        f"# {meta['title']}",
        "",
        f"{meta['description']} Dataset: {', '.join(dict.fromkeys(e['dataset'] for e in examples))}.",
        "",
        "[:octicons-arrow-left-24: All examples](index.md)",
        "",
    ]
    for group, (group_title, group_description) in GROUPS.items():
        in_group = [e for e in examples if e["group"] == group]
        if not in_group:
            continue
        lines += [f"## {group_title}", "", group_description, "", '<div class="grid cards" markdown>', ""]
        for example in in_group:
            lines += card(example)
        lines += ["</div>", ""]
    return "\n".join(lines)


def index_page(by_task):
    lines = [
        "# Examples",
        "",
        "End-to-end examples on real datasets, as notebooks: read them here, or run them on "
        "Google Colab with one click. Each one explains a single uncertainty technique, applied "
        "either **during training** or **after training** (post-processing).",
        "",
        "## Tasks",
        "",
    ]
    for task, meta in TASKS.items():
        if task in by_task:
            count = len(by_task[task])
            lines.append(f"- {meta['icon']} [{meta['title']}]({task}.md): {meta['description']} ({count} examples)")
    lines += ["", "## All examples", ""]
    for task, meta in TASKS.items():
        for example in sorted(by_task.get(task, []), key=lambda e: list(GROUPS).index(e["group"])):
            group_title = GROUPS[example["group"]][0]
            lines.append(f"- [{example['title']}]({example['doc_file']}) — *{meta['title']}, {group_title.lower()}*")
    lines.append("")
    return "\n".join(lines)


def generate_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    by_task = {}
    for path in sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*", "*", "*.ipynb"))):
        meta = notebook_meta(path)
        if meta["task"] not in TASKS or meta["group"] not in GROUPS:
            raise ValueError(f"Unexpected example location: {path}")
        by_task.setdefault(meta["task"], []).append(meta)
        with open(os.path.join(OUTPUT_DIR, meta["doc_file"]), "w", encoding="utf-8") as f:
            f.write(example_page(path, meta))
    for task, examples in by_task.items():
        with open(os.path.join(OUTPUT_DIR, f"{task}.md"), "w", encoding="utf-8") as f:
            f.write(task_page(task, examples))
    with open(os.path.join(OUTPUT_DIR, "index.md"), "w", encoding="utf-8") as f:
        f.write(index_page(by_task))
    return by_task


if __name__ == "__main__":
    for task, examples in generate_all().items():
        print(f"{task}: {', '.join(e['stem'] for e in examples)}")
