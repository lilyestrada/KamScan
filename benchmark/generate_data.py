#!/usr/bin/env python3
"""Synthetic data generator for KamScan benchmarks."""

import argparse
import os
import random

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic KamScan data")
    parser.add_argument("--num_rows", type=int, required=True, help="Number of k-mer rows")
    parser.add_argument("--num_patients", type=int, required=True, help="Total number of patients")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def generate_patient_ids(num_patients):
    """Generate patient IDs split evenly between normal and tumor groups."""
    n_normal = num_patients // 2
    n_tumor = num_patients - n_normal
    normal_ids = [f"normal_{i}" for i in range(1, n_normal + 1)]
    tumor_ids = [f"tumor_{i}" for i in range(1, n_tumor + 1)]
    return normal_ids, tumor_ids


def generate_matrix(filepath, num_rows, patient_ids, tumor_ids_set, rng, block_size=10000):
    """Write k-mer count matrix in blocks to limit memory."""
    num_patients = len(patient_ids)
    # Pre-select ~10% of rows that get a differential signal in tumor group
    differential_rows = set(rng.choice(num_rows, size=max(1, num_rows // 10), replace=False))
    # Pre-compute which columns are tumor
    tumor_mask = np.array([pid in tumor_ids_set for pid in patient_ids])

    with open(filepath, "w") as f:
        # Header
        f.write("kmer_id " + " ".join(patient_ids) + "\n")

        rows_written = 0
        while rows_written < num_rows:
            block = min(block_size, num_rows - rows_written)

            # Zero-inflated negative binomial: 30% zeros
            zero_mask = rng.random((block, num_patients)) < 0.3
            counts = rng.negative_binomial(n=2, p=0.01, size=(block, num_patients))
            counts[zero_mask] = 0

            # Apply 3-5x multiplier to tumor columns for differential rows
            for i in range(block):
                global_row = rows_written + i
                if global_row in differential_rows:
                    multiplier = rng.integers(3, 6)  # 3, 4, or 5
                    counts[i, tumor_mask] = counts[i, tumor_mask] * multiplier

            # Write rows
            lines = []
            for i in range(block):
                kmer_id = f"KMER_{rows_written + i + 1:07d}"
                row_str = " ".join(str(c) for c in counts[i])
                lines.append(f"{kmer_id} {row_str}\n")
            f.writelines(lines)

            rows_written += block


def generate_condition_file(filepath, normal_ids, tumor_ids, rng):
    """Generate shuffled patient-to-condition mapping."""
    entries = [(pid, "normal") for pid in normal_ids] + [(pid, "tumoral") for pid in tumor_ids]
    rng.shuffle(entries)
    with open(filepath, "w") as f:
        for pid, cond in entries:
            f.write(f"{pid} {cond}\n")


def generate_cpm_file(filepath, patient_ids, rng):
    """Generate CPM normalization file with random k-mer counts."""
    with open(filepath, "w") as f:
        f.write("File Nb_kmers\n")
        for pid in patient_ids:
            count = rng.integers(800_000_000, 2_800_000_001)
            f.write(f"{pid} {count}\n")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    conditions_dir = os.path.join(args.output_dir, "conditions")
    os.makedirs(conditions_dir, exist_ok=True)

    normal_ids, tumor_ids = generate_patient_ids(args.num_patients)
    # Shuffle patient column order for realism
    all_ids = normal_ids + tumor_ids
    py_rng.shuffle(all_ids)
    tumor_set = set(tumor_ids)

    print(f"Generating matrix: {args.num_rows} rows x {args.num_patients} patients...")
    generate_matrix(
        os.path.join(args.output_dir, "matrix.txt"),
        args.num_rows, all_ids, tumor_set, rng,
    )

    print("Generating condition file...")
    generate_condition_file(
        os.path.join(conditions_dir, "sampleshuf.train1.txt"),
        normal_ids, tumor_ids, rng,
    )

    print("Generating CPM normalization file...")
    generate_cpm_file(
        os.path.join(args.output_dir, "design_kmers_nb_per_patient"),
        all_ids, rng,
    )

    print(f"Done. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
