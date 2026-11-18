
import os
import argparse
import subprocess
import sys


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--partition",
        default="train"
    )

    args = parser.parse_args()

    base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    dataset = os.path.join(
        base, "Student folder", args.partition
    )

    scripts = [
        "gps_features.py",
        "candidate_generation.py",
        "prepare_ml_features.py"
    ]

    trips = sorted(
        name for name in os.listdir(dataset)
        if os.path.isdir(os.path.join(dataset, name))
    )

    print("====================================")
    print("PATHSNAP FEATURE ENGINEERING")
    print("Partition:", args.partition)
    print("Trips found:", len(trips))
    print("====================================")

    successful = []
    failed = []

    for index, trip in enumerate(trips, 1):

        print(f"\n[{index}/{len(trips)}] Processing {trip}")

        try:

            for script in scripts:

                command = [
                    sys.executable,
                    os.path.join(
                        base, "feature_engineering", script
                    ),
                    "--trip", trip
                ]

                if script == "candidate_generation.py":
                    command.extend([
                        "--partition", args.partition
                    ])

                subprocess.run(
                    command,
                    check=True,
                    cwd=base,
                    stdout=subprocess.DEVNULL
                )

            successful.append(trip)
            print("SUCCESS:", trip)

        except subprocess.CalledProcessError as error:

            failed.append(trip)
            print("FAILED:", trip, error)

    print("\n========== PIPELINE SUMMARY ==========")
    print("Successful trips:", len(successful))
    print("Failed trips:", len(failed))

    if failed:
        print("Failed trip IDs:", failed)

    print("Output directory:", os.path.join(base, "results"))


if __name__ == "__main__":
    main()
