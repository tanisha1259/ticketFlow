from analyzer import analyze_ticket


def main():
    ticket = """
    I was charged twice for my monthly subscription.
    The first payment went through successfully, but
    I was charged again a few minutes later.
    I have not received a refund and I need this
    resolved quickly.
    """

    result = analyze_ticket(ticket)

    print("\n========== TICKET ANALYSIS ==========\n")

    print(f"Category: {result.category}")
    print(f"Priority: {result.priority}")
    print(f"Sentiment: {result.sentiment}")

    print("\nSummary:")
    print(result.summary)

    print("\nRecommended Action:")
    print(result.recommended_action)

    print("\n=====================================")


if __name__ == "__main__":
    main()