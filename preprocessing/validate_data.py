
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

print("=" * 55)
print("       PATHSNAP DETAILED DATA VALIDATION")
print("=" * 55)

# Expected columns for each file type
expected_columns = {
    "track": ["longitude", "latitude", "timestamp"],
    "nodes": ["longitude", "latitude"],
    "arcs": ["from_node_id", "to_node_id"],
    "route": ["edge_id"]
}

total_files = 0
failed_files = []
empty_files = []
missing_value_files = []
non_numeric_files = []

# Check summary file
summary_file = DATA_DIR / "preprocessing_summary.csv"

if not summary_file.exists():
    raise FileNotFoundError("Preprocessing summary not found!")

summary = pd.read_csv(summary_file, dtype={"trip_id": str})

print("\n1. SUMMARY VALIDATION")
print("Total training trips:", len(summary))
print("Missing summary values:", summary.isnull().sum().sum())
print("Duplicate trip IDs:", summary["trip_id"].duplicated().sum())

# Check all processed CSV files
print("\n2. CHECKING PROCESSED CSV FILES")

for file_type, columns in expected_columns.items():

    files = sorted(DATA_DIR.glob(f"*_{file_type}.csv"))

    print(f"\nChecking {file_type}: {len(files)} files")

    for file in files:
        total_files += 1

        try:
            df = pd.read_csv(file)

            # Check column names
            if list(df.columns) != columns:
                failed_files.append(
                    (file.name, "Incorrect column names")
                )
                continue

            # Check empty files
            if df.empty:
                empty_files.append(file.name)
                continue

            # Check missing values
            if df.isnull().values.any():
                missing_value_files.append(file.name)

            # Check numeric values
            for column in columns:
                converted = pd.to_numeric(
                    df[column], errors="coerce"
                )

                if converted.isnull().any():
                    non_numeric_files.append(file.name)
                    break

        except Exception as error:
            failed_files.append((file.name, str(error)))

# Final results
print("\n" + "=" * 55)
print("3. FINAL VALIDATION RESULTS")
print("=" * 55)

print("Total processed CSV files checked:", total_files)
print("Files with incorrect columns or reading errors:", len(failed_files))
print("Empty files:", len(empty_files))
print("Files with missing values:", len(missing_value_files))
print("Files with non-numeric values:", len(non_numeric_files))

if failed_files:
    print("\nFILES WITH ERRORS:")
    for name, reason in failed_files[:20]:
        print(name, ":", reason)

if empty_files:
    print("\nEMPTY FILES:")
    for name in empty_files[:20]:
        print(name)

if missing_value_files:
    print("\nFILES WITH MISSING VALUES:")
    for name in missing_value_files[:20]:
        print(name)

if non_numeric_files:
    print("\nFILES WITH NON-NUMERIC VALUES:")
    for name in non_numeric_files[:20]:
        print(name)

if (
    not failed_files
    and not empty_files
    and not missing_value_files
    and not non_numeric_files
    and total_files == 320
):
    print("\nSUCCESS: ALL 320 CSV FILES PASSED VALIDATION!")
else:
    print("\nWARNING: SOME VALIDATION CHECKS FAILED.")

print("\nVALIDATION COMPLETED!")
