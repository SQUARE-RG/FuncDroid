# FuncDroid

FuncDroid is a functionality-oriented mobile GUI testing and exploration tool. Instead of optimizing for traditional coverage metrics alone, it aims to **model app functionalities and their interactions**, then use that model to guide exploration and generate higher-value test cases.

## Motivation

Modern mobile apps evolve quickly and contain many interacting features. In practice, testers still rely heavily on manual, **functionality-oriented** test design because it often finds bugs that coverage-driven exploration misses. Recent research uses large language models (LLMs) to add semantic understanding to GUI testing, but many existing approaches treat a "functionality" as a simple linear event sequence and often test features in isolation - making it hard to uncover deeper bugs caused by **interactions between functionalities**.

FuncDroid addresses this with an inter-functionality, state-aware model and a closed-loop testing workflow.

## Approach (IFO: Inter-Functionality Flow Oriented Testing)

### Overview

![Overview](overview.png)

### Functionality Flow Graph (FFG)

FuncDroid models an app as a directed **Functionality Flow Graph (FFG)**:
- Each node is a **meaningful functionality**: a complete, user-centric task that is logically indivisible (i.e., it does not contain other meaningful sub-functionalities).
- Each node can be characterized by `(goal, vars, traces)`:
  - `goal`: semantic intent (e.g., "Add alarm")
  - `vars`: state variables used/modified during execution (GUI/data/system-environment dimensions)
  - `traces`: observed execution traces (sequences of widget actions)
- Each edge captures an inter-functionality **flow** annotated with `(pi, phi, pi')`:
  - after executing trace `pi` of functionality `n`, if state condition `phi` holds, then trace `pi'` of functionality `n'` becomes executable
  - `phi` defines a meaningful subset of the app state space, so testing can focus on exploring those state subsets rather than blind exhaustive search

### Iterative, Evidence-Driven Testing Loop

FuncDroid follows an iterative workflow that (1) builds/updates the FFG from execution evidence and (2) generates targeted scenarios to refine the model and find deep interaction bugs:

1) **FFG initializing and updating**
   - Start with lightweight automated exploration to collect initial traces
   - Use an (M)LLM to infer page goals, segment traces into candidate meaningful functionalities, and extract state variables
   - Create initial flows from observed temporal succession (initially setting `phi = True` conservatively), then refine `phi` using new execution evidence
   - Update nodes by merging/splitting/creating functionalities based on semantic coherence and variable usage

2) **Long-short term view scenario testing**
   - Long-term view: generate scenarios to challenge and refine functionality definitions (completeness/independence) and flow preconditions (`phi`) via strategies like repetition, negation, and partition
   - Short-term view: generate metamorphic, data-driven scenarios by transforming GUI/data states at widget-level, intra-functionality, and inter-functionality layers
   - Scenario-driven bug detection: translate scenarios into executable traces, run them adaptively (with recovery), and log evidence for the next update cycle

## What This Repository Contains

- `funcdroid/`: core implementation:
  - `main.py`: CLI entrypoint for the defect-detection workflow
  - `workflow/`: LangGraph-based exploration loop (scenario planning/execution, FFG update, async bug detection, page audit)
  - `model/flow_graph.py`: Functionality Flow Graph structures
  - `automator/`, `connector/`, `device.py`: device automation layer (uiautomator2/HarmonyOS, ADB/HDC)
- `Effectiveness evaluation/`, `Usefulness evaluation/`: experiment outputs and datasets (large; not required for running the tool).

## Prerequisites

- Windows, macOS, or Linux with **Python 3.11, 3.12, or 3.13**
- An Android device (or emulator) with **USB debugging enabled**
- [Android Platform Tools](https://developer.android.com/tools/releases/platform-tools), with `adb` available in `PATH`
- One **OpenAI-compatible multimodal model endpoint**. The selected model must accept image inputs.

Java and Android Studio are not required. FuncDroid uses `adb` and `uiautomator2` to control an already installed app.

## Quick Start

### 1. Install FuncDroid

On macOS or Linux, run the setup script from the repository root:

```bash
chmod +x setup.sh run.sh
./setup.sh
```

The script creates `.venv`, installs the pinned dependencies, and creates `funcdroid/.env` if it does not exist. If Python 3.11-3.13 is missing and [`uv`](https://docs.astral.sh/uv/) is installed, the script can install Python 3.13 automatically.

On Windows PowerShell, use:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path funcdroid\.env)) { Copy-Item funcdroid\.env.example funcdroid\.env }
```

### 2. Configure the model

All model roles share a single OpenAI-compatible endpoint, configured via environment variables (loaded automatically by `python-dotenv`):

```bash
# macOS/Linux
nano funcdroid/.env

# Windows
notepad funcdroid\.env
```

Fill in these three values and leave the optional settings at their defaults:

```ini
API_KEY=your_api_key
BASE_URL=https://your-openai-compatible-endpoint/v1
MODEL=your_multimodal_model
```

`funcdroid/.env` is ignored by Git and must never be committed. Runtime knobs such as timeouts and exploration budgets are documented in `.env.example`.

### 3. Connect a device and run

Enable USB debugging, connect the device, approve the authorization prompt, and verify the connection:

```bash
adb devices
```

Open the app you want to test on the device. For a safe first check, run two iterations with a five-minute limit:

```bash
# macOS/Linux: one connected device is selected automatically
./run.sh --max-iterations 2 --total-time 5

# Windows PowerShell
.\.venv\Scripts\python.exe funcdroid\main.py --max-iterations 2 --total-time 5
```

The first iteration reads the initial screen and builds the Functionality Flow Graph. Later iterations plan and execute scenarios. When the short check succeeds, start a normal run with the default 90-minute budget:

```bash
./run.sh
```

FuncDroid tests the foreground app when `--package-name` is omitted. To launch and test a specific installed app, pass its package name:

```bash
./run.sh --package-name com.example.app --app-name "Example App"
```

When multiple devices are connected, select one explicitly:

```bash
./run.sh --device-id emulator-5554
```

Results are written under `funcdroid/output/`. Each run includes the FFG, execution traces, screenshots, logs, bug reports, LLM usage, and an artifact manifest.

## Common Commands

```bash
# Show every CLI option
./run.sh --help

# Run only intra-functionality scenarios
./run.sh --phase intra

# Disable the asynchronous bug-review worker
./run.sh --no-bug-detection

# Continue an interrupted run without discarding its FFG
./run.sh --resume-output output/<existing-run-directory>
```

## Troubleshooting

- `adb devices` shows `unauthorized`: unlock the device and approve its USB-debugging prompt.
- No device is detected: install Android Platform Tools, add `adb` to `PATH`, reconnect the cable, and run `adb devices` again.
- `uiautomator2` cannot connect: run `.venv/bin/python -m uiautomator2 -s <device-id> init` (Windows: `.venv\Scripts\python.exe ...`) and retry.
- LLM configuration is missing: confirm that `API_KEY`, `BASE_URL`, and `MODEL` are non-empty in `funcdroid/.env`.
- More than one device is connected: add `--device-id <serial>`; serials are listed by `adb devices`.
- A previous output directory is rejected: choose a new `--output-dir`, or use `--resume-output` only when intentionally continuing that run.

## Bug Showcase

FuncDroid has discovered a collection of real-world functional bugs in popular mobile applications. We provide a public bug showcase containing reviewed and reproducible evidence, including execution traces, screenshots, bug descriptions, and confirmation status.

[FuncDroid Bug Showcase](https://fim-bugs.sites.veilaxis.com/)

The showcase presents bugs discovered during automated exploration and preserves the corresponding evidence for reproduction and further analysis.

## Citation

If you use FuncDroid in your research or project, please consider citing:

1. Jinlong He, Changwei Xia, Binru Huang, Jiwei Yan, Jun Yan, Jian Zhang. FuncDroid: Towards Inter-Functional Flows for Comprehensive Mobile App GUI Testing. ISSTA 2026. [paper](https://arxiv.org/abs/2602.12834)

```bibtex
@inproceedings{FuncDroid,
      title={FuncDroid: Towards Inter-Functional Flows for Comprehensive Mobile App GUI Testing},
      author={Jinlong He and Changwei Xia and Binru Huang and Jiwei Yan and Jun Yan and Jian Zhang},
      booktitle={Proceedings of the 35th ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA)},
      year={2026}
}
```

## Maintainers and Contributors

FuncDroid is actively developed and maintained by members of [SQUARE Research Group](https://square16.org/):

- [Jinlong He](https://hanada31.github.io/)([@Jinlong-He](https://github.com/Jinlong-He))
- [Changwei Xia](https://hanada31.github.io/)([@daxiami233](https://github.com/daxiami233))
- [Binru Huang](https://hanada31.github.io/)([@ningxiudg](https://github.com/ningxiudg))
- [Jiwei Yan](https://hanada31.github.io/)([@hanada31](https://github.com/hanada31))

## Contributing

FuncDroid is an open-source research project, and contributions from the community are welcome.
