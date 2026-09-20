# TicketFlow

A local AI-powered customer support ticket analyzer built with **DSPy**, **Ollama**, and **Qwen3**.

TicketFlow takes a customer support ticket and produces:

* **Category** — Billing, Technical, Account, Subscription, or General
* **Priority** — Low, Medium, or High
* **Sentiment** — Positive, Neutral, or Negative
* **Summary** — concise description of the customer's request
* **Recommended Action** — suggested next step for the support team

The entire inference pipeline runs locally through Ollama.

---

## Architecture

```text
Customer Support Ticket
          │
          ▼
   DSPy TicketClassifier
          │
          ▼
 Classification Validation
          │
          ├── Category
          ├── Priority
          └── Sentiment
          │
          ▼
   DSPy TicketResponder
          │
          ├── Summary
          └── Recommended Action
          │
          ▼
      Final Analysis
```

### Model Stack

```text
Application
    │
    ▼
   DSPy
    │
    ▼
Ollama API
    │
    ▼
 Qwen3:4b
```

---

## Features

### Structured Classification

Tickets are classified into five categories:

* Billing
* Technical
* Account
* Subscription
* General

Priority is classified as:

* Low
* Medium
* High

Sentiment is classified as:

* Positive
* Neutral
* Negative

### Response Generation

After classification, TicketFlow generates:

* A concise ticket summary
* A recommended next action for the support team

For general informational requests, the responder is instructed not to invent company-specific information.

### Output Validation

Model output is validated before being returned by the application.

Invalid category, priority, or sentiment values raise an error instead of silently propagating malformed model output.

Response generation also includes validation against empty or placeholder outputs.

### Local Inference

The project uses:

* **Ollama** for local model serving
* **Qwen3 4B** as the language model
* **DSPy** for programmatic LLM orchestration

No external LLM API is required for inference.

### Batch Processing

TicketFlow can process multiple tickets from a JSON file and save the results as JSON.

Example input:

```json
[
  "I was charged twice for my subscription.",
  "I'd like to update the email linked to my profile.",
  "What are your customer support hours?"
]
```

Run:

```bash
uv run python -m ticketflow.batch tickets.json -o results.json
```

The batch processor supports both sequential and optional concurrent execution.

---

## Why DSPy?

Instead of manually constructing a large prompt, TicketFlow defines structured DSPy signatures for classification and response generation.

DSPy allows the application to define:

* Expected inputs
* Expected outputs
* Task instructions
* Reusable modules
* Optimization workflows

The project also includes a small DSPy `BootstrapFewShot` optimization experiment.

---

## Evaluation

The project uses separate development and held-out test examples.

The development set is used for experimentation and DSPy optimization. The held-out test set is used to evaluate the classifier.

The current held-out test set contains **11 examples** covering the supported categories and several common ticket types.

The latest evaluation produced:

```text
Score:    9/11
Accuracy: 81.8%
```

The evaluation set is intentionally small, so this result should be treated as a project-level validation result rather than a statistically representative benchmark.

The classifier was tested on categories including Billing, Technical, Account, Subscription, and General.

---

## DSPy Optimization Experiment

TicketFlow also contains an experimental `BootstrapFewShot` optimization workflow.

Run:

```bash
uv run python -m ticketflow.optimize
```

The resulting optimized classifier is saved as:

```text
optimized_classifier.json
```

The optimization experiment did **not** demonstrate a consistent improvement over the original classifier, so the optimized classifier is not used by the default application pipeline.

The artifact is retained as an experiment for future investigation.

---

## Requirements

* Python 3.11+
* [Ollama](https://ollama.com/)
* `uv`
* Qwen3 4B model

Pull the model with:

```bash
ollama pull qwen3:4b
```

Verify Ollama is running:

```bash
ollama list
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tanisha1259/ticketFlow.git
cd ticketFlow
```

Create the environment and install dependencies:

```bash
uv venv
uv sync
```

Activate the environment if desired.

### Git Bash

```bash
source .venv/Scripts/activate
```

---

## Running TicketFlow

Start the interactive application:

```bash
uv run python -m ticketflow.main
```

Example:

```text
========== TicketFlow ==========
Enter a customer support ticket.
Type 'exit' to quit.

Ticket: I was charged twice for my subscription.

---------- Analysis ----------
{
  "ticket": "I was charged twice for my subscription.",
  "category": "Billing",
  "priority": "High",
  "sentiment": "Negative",
  "summary": "Customer reports receiving two charges for their subscription.",
  "recommended_action": "Investigate payment records to identify duplicate charges and contact the customer to resolve the issue."
}
------------------------------
```

Type `exit` to stop the application.

---

## Running Batch Analysis

Create a JSON file containing a list of tickets:

```json
[
  "I was charged twice for my subscription.",
  "I'd like to update the email linked to my profile.",
  "What are your customer support hours?"
]
```

Then run:

```bash
uv run python -m ticketflow.batch tickets.json -o results.json
```

For optional concurrent processing:

```bash
uv run python -m ticketflow.batch tickets.json -o results.json --concurrent --workers 2
```

The generated `results.json` contains the structured analysis for each ticket.

Generated result files are excluded from version control through `.gitignore`.

---

## Running Evaluation

Run the held-out classifier evaluation:

```bash
uv run python -m ticketflow.evaluation
```

The evaluation reports individual predictions and the overall exact-match accuracy.

---

## Running the DSPy Optimization Experiment

```bash
uv run python -m ticketflow.optimize
```

This compiles the classifier using the development examples and saves:

```text
optimized_classifier.json
```

The optimized artifact is experimental and is not required to run the main application.

---

## Project Structure

```text
ticketFlow/
│
├── src/
│   └── ticketflow/
│       ├── __init__.py
│       ├── analyzer.py
│       ├── batch.py
│       ├── dataset.py
│       ├── evaluation.py
│       ├── main.py
│       └── optimize.py
│
├── tickets.json
├── optimized_classifier.json
├── .python-version
├── pyproject.toml
├── README.md
├── uv.lock
└── .gitignore
```

### File Responsibilities

**`analyzer.py`**

Contains the DSPy signatures and modules used for classification, response generation, validation, and the complete TicketFlow pipeline.

**`dataset.py`**

Contains separate development and held-out test examples.

**`evaluation.py`**

Runs the classifier against the held-out test set and calculates the exact-match evaluation score.

**`optimize.py`**

Runs the experimental DSPy `BootstrapFewShot` optimization.

**`batch.py`**

Loads tickets from a JSON file, processes them through TicketFlow, and saves structured results.

**`main.py`**

Provides the interactive command-line interface.

**`tickets.json`**

Provides a small example batch of customer support tickets.

---

## Future Extensions

Potential future improvements include:

* Retrieval from support documentation
* Tool-using support workflows
* API-based access to TicketFlow
* Evaluation on a larger real-world support-ticket dataset
* More robust response grounding
* Improved model serving and inference performance

These extensions are outside the current application. The current implementation focuses on a functional local DSPy → Ollama → Qwen3 support-ticket analysis pipeline.
