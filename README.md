# MLP for MNIST

This is my Homework X2 project for **Introduction to Deep Learning**. It extends the linear MNIST classifier from Homework 1 to an MLP with one ReLU hidden layer. The notebook implements initialization, the forward pass, stable softmax cross-entropy, backpropagation, prediction, and mini-batch SGD using NumPy.

The three eight-epoch experiments selected the wider hidden layer by final validation accuracy: **64 hidden units**, learning rate **0.10**, and L2 coefficient **0.0001**. Its training accuracy was **94.15%**, validation accuracy was **93.53%**, and one-time test accuracy was **93.89%**. The derivation and result analysis are in [the report](report/HWX2_Report.md); the [PDF report](report/HWX2_Report.pdf) is ready to read.

## Repository Layout

```text
GitHub/
├── README.md
├── requirements.txt
├── .gitignore
├── codes/
│   ├── HWX2_MLP_MNIST.ipynb
│   ├── mnist_data_loader.py
│   └── run_notebook.py
├── data/
│   └── mnist_data/
│       └── README.md
├── images/
│   ├── mnist_samples.png
│   ├── learning_curves.png
│   └── author_name.pdf
├── references/
│   ├── README.md
│   ├── HWX2/
│   └── HW1/
└── report/
    ├── HWX2_Report.md
    ├── HWX2_Report.tex
    ├── HWX2_Report.pdf
    ├── results.csv
    ├── epoch_results.csv
    └── experiment_record.json
```

The four MNIST `.gz` files are local inputs and are omitted from this package, as required by the assignment. The small README keeps their intended location visible.

| File or folder | Purpose |
| --- | --- |
| `codes/HWX2_MLP_MNIST.ipynb` | Completed notebook with all Python cells executed, checks, training output, inline figures, test result, and record export. |
| `codes/mnist_data_loader.py` | Unchanged course-supplied loader for compressed MNIST IDX files. |
| `codes/run_notebook.py` | Optional terminal runner that executes the notebook's Python cells and saves their text and figure outputs back into the notebook. |
| `data/mnist_data/` | Location for the four course-supplied dataset files before rerunning. |
| `images/mnist_samples.png` | Eight example images with labels. |
| `images/learning_curves.png` | Training loss and training/validation accuracy for all three runs. |
| `images/author_name.pdf` | Small vector name asset from my HW1 report, used so the English LaTeX report can render my Chinese name without a separate CJK font installation. |
| `report/HWX2_Report.md` | Report with Honor Code, derivation, settings, checks, results, discussion, and sources. |
| `report/HWX2_Report.tex` | Standalone XeLaTeX report source. |
| `report/HWX2_Report.pdf` | Compiled report. |
| `report/results.csv` | One final-results row per configuration; only the selected row has a test result. |
| `report/epoch_results.csv` | One row per epoch and configuration, 24 rows in total. |
| `report/experiment_record.json` | Settings, full histories, checks, selected model, test result, input checksums, split-index hashes, and execution environment. |
| `requirements.txt` | NumPy and Matplotlib versions used for the recorded run. |

## Requirements and Installation

The recorded experiments used **Python 3.12.14**, **NumPy 2.3.5**, and **Matplotlib 3.10.8** on Linux. These are the versions present during training and export. Python 3.11 or 3.12 can be used with the pinned dependencies; small floating-point differences may occur across platforms and BLAS libraries.

From this project root, install:

```bash
python -m pip install -r requirements.txt
```

On Windows, `py -3.12 -m pip install -r requirements.txt` selects Python 3.12 explicitly. For a separate environment in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If using VS Code, install the Microsoft Python and Jupyter extensions and select this same environment as the notebook kernel. If a kernel is missing, install it in that environment:

```bash
python -m pip install ipykernel
```

The terminal runner does not require Jupyter or ipykernel. The MLP uses NumPy and Matplotlib with the Python standard library. It does not use automatic differentiation or prebuilt model, softmax, loss, or optimizer routines.

## Prepare the Data

Copy the four files from the original HWX2 course package into `data/mnist_data/`:

| Filename | Contents |
| --- | --- |
| `train-images-idx3-ubyte.gz` | 60,000 original training images |
| `train-labels-idx1-ubyte.gz` | Their labels |
| `t10k-images-idx3-ubyte.gz` | 10,000 separate test images |
| `t10k-labels-idx1-ubyte.gz` | Their labels |

Keep the files compressed and keep their exact names. The loader reads gzip files directly. No internet connection or automatic dataset download is needed.

## Two Supported Data Layouts

| Layout | Notebook and loader | Dataset | Outputs |
| --- | --- | --- | --- |
| Project layout | Root `codes/` | Root `data/mnist_data/` | Root `images/` and `report/` |
| Original assignment layout | Both directly in one folder | `mnist_data/` beside them | `images/` and `report/` in that folder |

For notebook execution, start the kernel in this project root or its `codes/` folder. In the original layout, start in the folder containing the notebook and loader. The path cell checks only this root and does not search unrelated parent projects. It imports the loader after adding `CODE_DIR` to `sys.path` and calls `load_mnist(DATA_DIR)`. The loader itself is unchanged.

To inspect a notebook kernel's working directory:

```python
from pathlib import Path
print(Path.cwd())
```

## Run the Notebook

The saved notebook already contains the completed experiment outputs. Reading the notebook and report does not require installing packages or restoring the dataset. The four data files are needed only to rerun the cells.

### Method 1: VS Code or Jupyter

1. Open this project root in VS Code and open `codes/HWX2_MLP_MNIST.ipynb`.
2. Select the environment containing NumPy, Matplotlib, and ipykernel.
3. Put the four data files in the required folder.
4. Restart the kernel and run every cell from top to bottom.
5. Confirm the split shapes are `(50000, 784)`, `(10000, 784)`, and `(10000, 784)`.
6. Confirm `All checks passed.` and `ReLU mask checks passed.` appear.
7. Wait for all three eight-epoch runs. Selection uses their final validation accuracies.
8. The plotting cells save the two PNG figures. The final test cell evaluates only the selected model.
9. The last cell exports the three record files without repeating test evaluation.
10. Save the notebook with its outputs.

The test cell rejects a second evaluation in the same kernel. To read its result again, use its saved output or `experiment_record.json`. If starting a new full experiment, restart the kernel first. The reporting and export cells can read the existing variables without another test evaluation.

### Method 2: Terminal

From the project root:

```bash
python codes/run_notebook.py
```

Or run the same command with the environment's Python executable:

```powershell
.\.venv\Scripts\python.exe codes\run_notebook.py
```

The runner executes the actual notebook source in cell order in a fresh Python process, captures its printed output and Matplotlib figures, and writes the executed `.ipynb`. It runs all training and the one selected-model test evaluation. This was the execution method used for the included notebook; no training output was filled in by hand. NumPy and Matplotlib are the only third-party packages required for this method.

Each method overwrites the same-named figures and record files. The terminal method also overwrites the notebook outputs. Neither method rewrites the Markdown/LaTeX/PDF report or recompiles it; if experimenting with other settings, update the report from the new matching records.

## Experiment Settings and Reproducibility

| Configuration | Hidden width | Learning rate | L2 coefficient | Epochs | Batch size |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 32 | 0.10 | 0.0001 | 8 | 256 |
| smaller learning rate | 32 | 0.03 | 0.0001 | 8 | 256 |
| wider hidden layer | 64 | 0.10 | 0.0001 | 8 | 256 |

All runs use seed **2026** for initialization and mini-batch shuffling. The split is made once with `default_rng(2026).permutation(60000)`: the first 50,000 indices are training examples and the remaining 10,000 are validation examples. Images are flattened and scaled to `[0, 1]` using `float32`.

Weights use Gaussian standard deviations `sqrt(2 / 784)` and `sqrt(1 / H)`, and biases start at zero. The initialization is reset for every run. The smaller-learning-rate run has exactly the same initial model and mini-batch order as baseline. The width comparison keeps the other settings fixed, with the initialization distribution adjusted for the new width.

The objective is mean cross-entropy plus `0.5 * reg * (sum(W1**2) + sum(W2**2))`. Biases are not regularized, and the penalty is not divided by batch size. Every update uses gradients computed before any parameter is changed. Only training examples enter updates.

Training loss is evaluated over the complete training set at each epoch's end, rather than averaged from losses observed at different points during an epoch. The selected model is the final-epoch model with the highest validation accuracy; the first listed configuration wins an exact tie. There is no early stopping, learning-rate schedule, or retraining on validation data.

The included run used `OPENBLAS_NUM_THREADS=4` and `OMP_NUM_THREADS=4`. To repeat these settings in PowerShell before terminal execution:

```powershell
$env:OPENBLAS_NUM_THREADS = "4"
$env:OMP_NUM_THREADS = "4"
python codes/run_notebook.py
```

Small numerical differences can still occur across environments. The recorded metrics belong to the included executed notebook and its matching figures.

## Saved Experiment Records

Accuracies in the CSV and JSON files are fractions: `0.9389` means `93.89%`. Only the selected configuration has a test-accuracy entry in `results.csv`. The 24 rows of `epoch_results.csv` contain the three runs' full eight-epoch histories. The JSON also records the exact selection rule and a count of one test evaluation.

The export timestamp is in UTC. Environment versions were collected in the same uninterrupted execution as training. SHA-256 values identify the supplied compressed data and fixed split indices. The record files do not contain dataset examples or model weights.

Running the final export cell again only reads the results in memory and overwrites the three records. If the kernel has restarted, rerun the preceding cells first; saved output text does not restore Python variables.

## Changes to the Supplied Notebook

- Completed all five required implementation blocks.
- Kept the supplied parameter shapes, objective, split, seed, configurations, and implementation checks.
- Used eight epochs for every configuration instead of the starter's three.
- Added path handling for the project and original layouts, retaining the supplied loader unchanged.
- Added example-image and learning-curve export with white backgrounds.
- Added finite-difference checks with inactive ReLU units and a direct check of the derivative convention at zero.
- Added a repeated-test guard and export of summary results, full epoch histories, checksums, and environment details.
- Added an optional terminal runner for the same notebook cells.

## Report and Figure Paths

The report sources are in `report/` and the figures are in the sibling `images/` folder. Keep the relative paths `../images/...` when moving or compiling the report. Compile from `report/` using XeLaTeX twice:

```bash
xelatex -interaction=nonstopmode -halt-on-error HWX2_Report.tex
xelatex -interaction=nonstopmode -halt-on-error HWX2_Report.tex
```

The source uses a normal English article layout and a vector asset for my Chinese name. It does not require an external preamble or a system CJK font. The included PDF was compiled and its pages were visually checked.

## Sources and AI Assistance

The course supplied the assignment PDF, starter notebook, loader, and dataset. The HW1 comparison uses my previous report and saved record. ChatGPT helped with derivation, implementation, experiment execution, record export, report drafting, and LaTeX/PDF preparation. The report's Honor Code describes this assistance.

## Reference Files

The [references folder](references/README.md) includes the original course instructions, starter notebook, unchanged loader, and my previous HW1 report and record. The completed notebook, report, figures, and experiment records are the same as in the submission version. Only repository documentation, references, and `.gitignore` are added here.

## Git Ignore Rules

`.gitignore` excludes local datasets, virtual environments, Python caches, notebook checkpoints, and LaTeX temporary files. Keep the executed notebook, figures, report, CSV files, and JSON record tracked. The four MNIST files are omitted from this GitHub package as well.
