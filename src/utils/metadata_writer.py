import csv
import os


def write_metadata_csv(rows, csv_path):
    if not rows:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
