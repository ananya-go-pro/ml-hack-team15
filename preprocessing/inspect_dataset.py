
from pathlib import Path
from collections import Counter

# Dataset folder
DATA_DIR = Path("data/raw")

print("=" * 55)
print("          PATHSNAP DATASET INSPECTION")
print("=" * 55)

# Check whether dataset folder exists
if not DATA_DIR.exists():
    print("\nERROR: data/raw folder not found!")
    print("Please check whether the ZIP was extracted correctly.")
    exit()

# Find all files
all_files = [f for f in DATA_DIR.rglob("*") if f.is_file()]

# Count file extensions
extensions = Counter(f.suffix.lower() for f in all_files)

print("\n1. DATASET FILE COUNTS")
print("-" * 40)

for ext, count in sorted(extensions.items()):
    print(f"{ext}: {count}")

# Count trips in each partition
print("\n2. TRIP COUNTS")
print("-" * 40)

for partition in ["train", "student_test"]:
    folder = DATA_DIR / "Student folder" / partition

    # Also support datasets where partitions are directly under raw/
    if not folder.exists():
        folder = DATA_DIR / partition

    if folder.exists():
        trips = [f for f in folder.iterdir() if f.is_dir()]
        print(f"{partition}: {len(trips)} trips")
    else:
        print(f"{partition}: folder not found")

# Inspect sample files
print("\n3. SAMPLE FILE CONTENTS")
print("-" * 40)

for ext in [".track", ".nodes", ".arcs", ".route"]:

    sample = next(
        (f for f in all_files if f.suffix.lower() == ext),
        None
    )

    if sample is None:
        print(f"\n{ext}: NOT FOUND")
        continue

    print(f"\nFILE TYPE: {ext}")
    print("Filename:", sample.name)

    try:
        with open(sample, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if i >= 5:
                    break
                print(line.rstrip())

    except Exception as error:
        print("Reading error:", error)

# Check empty files
print("\n4. EMPTY FILE CHECK")
print("-" * 40)

empty_files = [f for f in all_files if f.stat().st_size == 0]

print("Total empty files:", len(empty_files))

for f in empty_files[:10]:
    print(f)

print("\n" + "=" * 55)
print("       DATASET INSPECTION COMPLETED")
print("=" * 55)
