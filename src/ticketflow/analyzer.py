import dspy


lm = dspy.LM(
    "ollama_chat/qwen3:4b",
    api_base="http://localhost:11434",
    api_key="ollama",
    model_type="chat",
    think=False,
)

dspy.configure(lm=lm)


class TicketAnalysis(dspy.Signature):
    """Analyze a customer support ticket.

    Categories:
    - Billing: payments, charges, refunds, invoices
    - Technical: bugs, crashes, errors, broken functionality
    - Account: profile, login, email, account settings
    - Subscription: subscribing, cancelling, changing subscription"""

    ticket: str = dspy.InputField(
        desc="The customer's support request."
    )

    category: str = dspy.OutputField(
        desc=(
            "The ticket category. Choose exactly one of: "
            "Billing, Technical, Account, Subscription."
        )
    )

    priority: str = dspy.OutputField(
        desc=(
            "Choose exactly one: Low, Medium, or High. "
            "Use High for financial loss, security issues, "
            "or major service disruption. "
            "Use Medium when normal functionality is affected. "
            "Use Low for informational or non-urgent requests."
        )
    )

    sentiment: str = dspy.OutputField(
        desc="Choose exactly one: Positive, Neutral, or Negative."
    )

    summary: str = dspy.OutputField(
        desc="A concise summary of the issue."
    )


class TicketAnalyzer(dspy.Module):

    def __init__(self):
        super().__init__()

        self.analyze = dspy.Predict(TicketAnalysis)

    def forward(self, ticket):

        return self.analyze(ticket=ticket)


analyzer = TicketAnalyzer()


def analyze_ticket(ticket: str):

    return analyzer(ticket=ticket)