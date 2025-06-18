# CI & Experiment Workflow Overview

This document outlines the two GitHub Actions workflows used in this project:


## ✅ 1. Run Tests – Continuous Integration (CI)

This workflow ensures the quality and correctness of the codebase by running tests on both architectural components and dataset-related modules. It runs automatically on:
	•	Pushes to main or develop
	•	Any pull request

### 🧪 Workflow Breakdown

The workflow consists of three jobs:
	•	Setup
Prepares the environment by:
	•	Checking out the repository
	•	Setting up Python
	•	Caching pip dependencies
	•	Test Architectures
	•	Depends on setup
	•	Installs dependencies
	•	Runs tests in tests/architectures/
	•	Test Datasets
	•	Depends on setup
	•	Installs dependencies
	•	Runs tests in tests/datasets/

### ⚙️ CI Workflow Diagram

flowchart TD
    subgraph Setup
      A1["Checkout repository"]
      A2["Setup Python"]
      A3["Cache pip"]
    end

    subgraph Test_Architectures
      B1["Checkout repository"]
      B2["Setup Python"]
      B3["Restore pip cache"]
      B4["Install dependencies"]
      B5["Run architecture tests"]
    end

    subgraph Test_Datasets
      C1["Checkout repository"]
      C2["Setup Python"]
      C3["Restore pip cache"]
      C4["Install dependencies"]
      C5["Run dataset tests"]
    end

    A1 --> A2 --> A3
    A3 --> B1 --> B2 --> B3 --> B4 --> B5
    A3 --> C1 --> C2 --> C3 --> C4 --> C5



## 🚀 2. Run Experiment Workflow

This workflow runs automatically on every push to main. It is designed to automate the process of configuring and executing experiments.

### 🔁 Workflow Steps
	•	Checkout Repository
Pulls the latest code.
	•	Set up Python
Uses Python 3.12.
	•	Install Dependencies
Installs all Python requirements.
	•	Generate Experiment
Runs a Python script that generates experiment configuration based on an args.json file.
	•	Run Experiment
Navigates to the experiment folder and executes the experiment via execute.sh.

### ⚙️ Experiment Workflow Diagram

flowchart TD
    subgraph Run_Experiment
      E1["Checkout repository"]
      E2["Setup Python 3.12"]
      E3["Install dependencies"]
      E4["Generate experiment config (generate.py)"]
      E5["Run experiment (execute.sh)"]
    end

    E1 --> E2 --> E3 --> E4 --> E5
