from analyzer import analyze_ticket


def main():
    print("\n========== TicketFlow ==========")
    print("Enter a customer support ticket.")
    print("Type 'exit' to quit.\n")

    while True:
        ticket = input("Ticket: ").strip()

        if ticket.lower() == "exit":
            print("\nGoodbye!")
            break

        if not ticket:
            print("Please enter a ticket.\n")
            continue

        try:
            result = analyze_ticket(ticket)

            print("\n---------- Analysis ----------")
            print(f"Category: {result.category}")
            print(f"Priority: {result.priority}")
            print(f"Sentiment: {result.sentiment}")

            print("\nSummary:")
            print(result.summary)

            print("\nRecommended Action:")
            print(result.recommended_action)

            print("------------------------------\n")

        except ValueError as error:
            print(f"\nUnable to analyze ticket: {error}\n")


if __name__ == "__main__":
    main()