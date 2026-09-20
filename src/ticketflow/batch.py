import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .analyzer import analyze_ticket


def analyze_tickets_sequential(tickets):
    """Analyze tickets one at a time."""
    results = []

    start = time.perf_counter()

    for ticket in tickets:
        try:
            results.append(analyze_ticket(ticket))
        except Exception as error:
            results.append({
                "ticket": ticket,
                "error": str(error),
            })

    elapsed = time.perf_counter() - start

    return results, elapsed


def analyze_tickets_concurrent(tickets, max_workers=2):
    """Analyze multiple tickets concurrently."""
    results = [None] * len(tickets)

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_ticket, ticket): index
            for index, ticket in enumerate(tickets)
        }

        for future in as_completed(futures):
            index = futures[future]

            try:
                results[index] = future.result()
            except Exception as error:
                results[index] = {
                    "ticket": tickets[index],
                    "error": str(error),
                }

    elapsed = time.perf_counter() - start

    return results, elapsed


def result_to_dict(result):
    """Convert a TicketResult or error dictionary into JSON-ready data."""
    if isinstance(result, dict):
        return result

    return result.to_dict()


def load_tickets(input_path):
    """Load tickets from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of tickets.")

    if not all(isinstance(ticket, str) for ticket in data):
        raise ValueError("Every ticket in the input must be a string.")

    return data


def save_results(results, output_path):
    """Save analysis results to a JSON file."""
    serializable_results = [
        result_to_dict(result)
        for result in results
    ]

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(serializable_results, file, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze customer support tickets with TicketFlow."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to a JSON file containing a list of tickets.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Path for the output JSON file.",
    )

    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Process tickets concurrently.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent workers.",
    )

    args = parser.parse_args()

    tickets = load_tickets(args.input)

    print(f"Loaded {len(tickets)} tickets.")

    if args.concurrent:
        results, elapsed = analyze_tickets_concurrent(
            tickets,
            max_workers=args.workers,
        )
        mode = f"concurrent ({args.workers} workers)"
    else:
        results, elapsed = analyze_tickets_sequential(tickets)
        mode = "sequential"

    output_path = save_results(results, args.output)

    print(f"Processing mode: {mode}")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()