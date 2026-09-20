from .analyzer import analyze_ticket


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
            print(result.to_json())
            print("------------------------------\n")

        except ValueError as error:
            print(f"\nUnable to analyze ticket: {error}\n")


if __name__ == "__main__":
    main()