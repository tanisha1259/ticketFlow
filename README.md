# TicketFlow

A local AI-powered customer support ticket analyzer built with **DSPy**, **Ollama**, and **Qwen3**.

TicketFlow takes a customer support ticket and produces:

* **Category** — Billing, Technical, Account, or Subscription
* **Priority** — Low, Medium, or High
* **Sentiment** — Positive, Neutral, or Negative
* **Summary** — concise description of the issue
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

### Model stack

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

## Why DSPy?

Instead of manually constructing a large prompt, TicketFlow defines structured DSPy signatures for classification and response generation.

DSPy manages the interaction between the program and the language model while allowing the application to define:

* expected inputs
* expected outputs
* task instructions
* reusable modules
* optimization workflows

The project also includes a small DSPy optimization experiment using `BootstrapFewShot`.

---

## Features

### Structured classification

Tickets are classified into four categories:

* Billing
* Technical
* Account
* Subscription

Priority is classified as:

* Low
* Medium
* High

Sentiment is classified as:

* Positive
* Neutral
* Negative

### Output validation

Model output is validated before it enters the response-generation stage.

Invalid category, priority, or sentiment values raise an error instead of silently propagating malformed model output.

Common formatting artifacts are also normalized before validation.

### Local inference

The project uses:

* **Ollama** for local model serving
* **Qwen3 4B** as the language model
* **DSPy** for programmatic LLM orchestration

No external LLM API is required for inference.

---

## Evaluation

The project uses separate development and held-out test examples.

The development set is used for experimentation and DSPy optimization.

The held-out test set contains 8 paraphrased examples that are not used for optimization.

After improving the classification instructions, the current held-out evaluation produced:

```text
Score:    7/8
Accuracy: 87.5%
```

The remaining error was a priority distinction for an application upload failure. Category and sentiment were correctly predicted for that case.

Because the evaluation set is intentionally small, the 87.5% result should be treated as a project-level validation result rather than a statistically representative benchmark.

---

## DSPy Optimization Experiment

TicketFlow also contains an experimental `BootstrapFewShot` optimization workflow.

```bash
uv run python src/ticketflow/optimize.py
```

The resulting optimized classifier is saved as:

```text
optimized_classifier.json
```

The experiment did **not** demonstrate a consistent improvement over the original classifier, so the optimized classifier is not used as the default production pipeline.

This keeps the main application based on the simpler, explicitly validated classifier while retaining the optimization experiment for future work.

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

Clone the repository and enter the project directory:

```bash
git clone https://github.com/tanisha1259/ticketFlow.git
cd ticketFlow
```

Create the environment and install dependencies:

```bash
uv venv
uv sync
```

Activate the environment if desired:

### Git Bash

```bash
source .venv/Scripts/activate
```

---

## Running TicketFlow

Start the interactive application:

```bash
uv run python src/ticketflow/main.py
```

Example:

```text
========== TicketFlow ==========
Enter a customer support ticket.
Type 'exit' to quit.

Ticket: I was charged twice for my subscription.

---------- Analysis ----------
Category: Billing
Priority: High
Sentiment: Negative

Summary:
Customer reports being charged twice for their subscription.

Recommended Action:
Investigate the duplicate transaction and initiate the appropriate
refund process.
------------------------------
```

Type `exit` to stop the application.

---

## Running Evaluation

Run the held-out classifier evaluation:

```bash
uv run python src/ticketflow/evaluation.py
```

The evaluation reports individual predictions and the overall exact-match accuracy.

---

## Running the DSPy Optimization Experiment

```bash
uv run python src/ticketflow/optimize.py
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
│       ├── dataset.py
│       ├── evaluation.py
│       ├── main.py
│       └── optimize.py
│
├── optimized_classifier.json
├── .python-version
├── pyproject.toml
├── README.md
├── uv.lock
└── .gitignore
```

### File responsibilities

**`analyzer.py`**

Contains the DSPy signatures and modules used for classification, response generation, validation, and the complete TicketFlow pipeline.

**`dataset.py`**

Contains separate development and held-out test examples.

**`evaluation.py`**

Runs the classifier against the held-out test set.

**`optimize.py`**

Runs the experimental DSPy `BootstrapFewShot` optimization.

**`main.py`**

Provides the interactive command-line interface.

---

## Future Extensions

Potential next steps include:

* Multi-stage support-ticket workflows
* Tool-using support agents
* LangGraph-based agent orchestration
* Retrieval from support documentation
* Batch ticket processing
* Structured JSON/API output
* Model serving with vLLM or SGLang
* Quantized model experiments
* Evaluation on a larger real-world support-ticket dataset

These extensions are intentionally outside the current MVP. The current implementation focuses on a reliable local DSPy → Ollama → Qwen3 pipeline.
