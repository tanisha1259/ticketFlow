import dspy
from ticketflow.analyzer import TicketAnalyzer

examples = [
    dspy.Example(
        ticket="I was charged twice for my subscription.",
        category="Billing",
        priority="High",
        sentiment="Negative",
        summary="Customer reports a duplicate subscription charge.",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="The application crashes every time I try to upload a file.",
        category="Technical",
        priority="High",
        sentiment="Negative",
        summary="Customer reports an application crash during file upload.",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="How can I change the email address associated with my account?",
        category="Account",
        priority="Low",
        sentiment="Neutral",
        summary="Customer wants to change their account email address.",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="I'd like to know how to cancel my subscription.",
        category="Subscription",
        priority="Low",
        sentiment="Neutral",
        summary="Customer wants instructions for cancelling their subscription.",
    ).with_inputs("ticket"),
]

def ticket_metric(example, prediction, trace=None):

    category_correct = (
        example.category.lower()
        == prediction.category.lower()
    )

    priority_correct = (
        example.priority.lower()
        == prediction.priority.lower()
    )

    sentiment_correct = (
        example.sentiment.lower()
        == prediction.sentiment.lower()
    )

    return (
        category_correct
        and priority_correct
        and sentiment_correct
    )


def evaluate():

    analyzer = TicketAnalyzer()

    for i, example in enumerate(examples, start=1):

        prediction = analyzer(
            ticket=example.ticket
        )

        print(f"\n{'=' * 60}")
        print(f"TEST CASE {i}")
        print(f"{'=' * 60}")

        print("\nTicket:")
        print(example.ticket)

        print("\nExpected:")
        print(f"Category:  {example.category}")
        print(f"Priority:  {example.priority}")
        print(f"Sentiment: {example.sentiment}")

        print("\nPredicted:")
        print(f"Category:  {prediction.category}")
        print(f"Priority:  {prediction.priority}")
        print(f"Sentiment: {prediction.sentiment}")

        print("\nMetric:")
        print(
            "PASS"
            if ticket_metric(example, prediction)
            else "FAIL"
        )


if __name__ == "__main__":
    evaluate()