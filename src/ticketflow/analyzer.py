import dspy


# -----------------------------
# Ollama + Qwen3 configuration
# -----------------------------

lm = dspy.LM(
    "ollama_chat/qwen3:4b",
    api_base="http://localhost:11434",
    api_key="ollama",
    model_type="chat",
    think=False,
)

dspy.configure(lm=lm)


# -----------------------------
# Ticket Classification
# -----------------------------

class TicketClassification(dspy.Signature):
    """
    Classify a customer support ticket.

    Categories:
    - Billing: payments, charges, refunds, invoices
    - Technical: bugs, crashes, errors, broken functionality
    - Account: profile, login, email, account settings
    - Subscription: subscribing, cancelling, changing subscription
    """

    ticket: str = dspy.InputField(
        desc="The customer's support request."
    )

    category: str = dspy.OutputField(
        desc=(
            "Choose exactly one: Billing, Technical, "
            "Account, or Subscription."
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


class TicketClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(TicketClassification)

    def forward(self, ticket):
        return self.classify(ticket=ticket)


# -----------------------------
# Ticket Response
# -----------------------------

class TicketResponseSignature(dspy.Signature):
    """Generate a useful response plan for a support ticket."""

    ticket: str = dspy.InputField(
        desc="The customer's support request."
    )

    category: str = dspy.InputField(
        desc="The ticket category."
    )

    priority: str = dspy.InputField(
        desc="The ticket priority."
    )

    summary: str = dspy.OutputField(
        desc="A concise summary of the customer's issue."
    )

    recommended_action: str = dspy.OutputField(
        desc="The recommended next action for the support team."
    )


class TicketResponder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(TicketResponseSignature)

    def forward(self, ticket, category, priority):
        return self.respond(
            ticket=ticket,
            category=category,
            priority=priority,
        )


# -----------------------------
# Complete TicketFlow pipeline
# -----------------------------

class TicketAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()

        self.classifier = TicketClassifier()
        self.responder = TicketResponder()

    def forward(self, ticket):
        classification = self.classifier(ticket)

        response = self.responder(
            ticket=ticket,
            category=classification.category,
            priority=classification.priority,
        )

        return dspy.Prediction(
            category=classification.category,
            priority=classification.priority,
            sentiment=classification.sentiment,
            summary=response.summary,
            recommended_action=response.recommended_action,
        )


analyzer = TicketAnalyzer()


def analyze_ticket(ticket: str):
    return analyzer(ticket=ticket)