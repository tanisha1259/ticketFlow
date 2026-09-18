import dspy


lm = dspy.LM(
    "ollama_chat/qwen3:4b",
    api_base="http://localhost:11434",
    api_key="ollama",
    model_type="chat",
    think=False
)

dspy.configure(lm=lm)


class TicketAnalysis(dspy.Signature):
    """Analyze a customer support ticket."""

    ticket: str = dspy.InputField(
        desc="The customer's support request."
    )

    category: str = dspy.OutputField(
        desc="The main category of the ticket."
    )

    priority: str = dspy.OutputField(
        desc="Priority: Low, Medium, or High."
    )

    sentiment: str = dspy.OutputField(
        desc="Customer sentiment: Positive, Neutral, or Negative."
    )

    summary: str = dspy.OutputField(
        desc="A concise summary of the issue."
    )


def analyze_ticket(ticket: str):

    analyzer = dspy.Predict(TicketAnalysis)

    return analyzer(ticket=ticket)