# CI Workflow Overview

This diagram illustrates the continuous integration (CI) workflow for running tests in this project. The workflow consists of three main jobs:
- Setup: Prepares the environment by checking out the repository, setting up Python, caching pip dependencies, and installing required packages.
- Test Architectures: Runs tests related to architectural components of the codebase.
- Test Datasets: Runs tests related to dataset handling and processing.

The architecture and dataset tests both depend on the successful completion of the setup job, ensuring dependencies are installed and cached before running.

```mermaid
flowchart TD
    subgraph Setup
      A1["Checkout repository"]
      A2["Setup Python"]
      A3["Cache pip"]
      A4["Install dependencies"]
    end

    subgraph Test_Architectures
      B1["Checkout repository"]
      B2["Setup Python"]
      B3["Run architecture tests"]
    end

    subgraph Test_Datasets
      C1["Checkout repository"]
      C2["Setup Python"]
      C3["Run dataset tests"]
    end

    A1 --> A2 --> A3 --> A4
    A4 --> B1 --> B2 --> B3
    A4 --> C1 --> C2 --> C3
```
