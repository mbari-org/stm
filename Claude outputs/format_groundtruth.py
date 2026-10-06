"""
format_groundtruth.py

Turns a Raven-style selection table (like
MARS-20231128T150000Z_1_32k.selections.groundtruth.txt) into the two things
torch_linear_model.py needs:

  1. A feature matrix X  -- one row per call, 6 numeric columns
     (torch_linear_model.py's `vector_dim = 6`)
  2. A list of human-readable label dicts -- one dict per call, whose keys
     match torch_linear_model.py's CATEGORY_NAMES, e.g.:
         {"unit_dc": "H", "code_fo": "11122", "unit_fo": "1"}
     This is exactly the format encode_labels() in torch_linear_model.py
     expects as input.

It also prints out a ready-to-paste CATEGORY_OPTIONS dict, since
torch_linear_model.py needs to know the full list of possible values for
each category up front (so it can size each linear "head" correctly).

Usage:
    python format_groundtruth.py path/to/groundtruth.txt [output_dir]
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Which raw columns in the Raven file become classification "categories"
# (i.e. which columns feed the multi-head classifier). Renamed here to
# valid Python identifiers so they can be used as CATEGORY_NAMES directly.
LABEL_COLUMNS = {
    "Unit (DC)": "unit_dc",
    "Code (FO)": "code_fo",
    "Unit (FO)": "unit_fo",
}

# Numeric columns that become the 6-dim input feature vector per call.
# (5 raw columns + a computed "Duration" column = 6, matching vector_dim=6)
FEATURE_COLUMNS = [
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "File Offset (s)",
]


def format_groundtruth(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="\t")

    # Duration is the 6th numeric feature.
    df["Duration (s)"] = df["End Time (s)"] - df["Begin Time (s)"]

    # A few rows in the raw file are missing a label (e.g. blank "Unit (DC)").
    # Those rows can't be used for supervised training, so drop them.
    label_cols_raw = list(LABEL_COLUMNS.keys())
    before = len(df)
    df = df.dropna(subset=label_cols_raw)
    df = df[(df[label_cols_raw].astype(str).apply(lambda c: c.str.strip()) != "").all(axis=1)]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with missing label values.")

    # --- Feature matrix ---
    feature_cols = FEATURE_COLUMNS + ["Duration (s)"]
    X = df[feature_cols].astype(float).values
    np.save(output_dir / "features.npy", X)
    pd.DataFrame(X, columns=feature_cols).to_csv(output_dir / "features.csv", index=False)

    # --- Human-readable labels (input to torch_linear_model.encode_labels) ---
    human_labels = []
    for _, row in df.iterrows():
        human_labels.append({
            new_name: str(row[raw_name]).strip()
            for raw_name, new_name in LABEL_COLUMNS.items()
        })
    with open(output_dir / "labels.json", "w") as f:
        json.dump(human_labels, f, indent=2)

    # --- CATEGORY_OPTIONS block to paste into torch_linear_model.py ---
    category_options = {
        new_name: sorted(df[raw_name].astype(str).str.strip().unique())
        for raw_name, new_name in LABEL_COLUMNS.items()
    }
    with open(output_dir / "category_options.py", "w") as f:
        f.write("CATEGORY_OPTIONS = {\n")
        for name, options in category_options.items():
            f.write(f"    {name!r}: {options!r},\n")
        f.write("}\n")

    print(f"Formatted {len(df)} examples.")
    print(f"Feature matrix shape: {X.shape}")
    for name, options in category_options.items():
        print(f"  {name}: {len(options)} option(s) -> {options}")
    print(
        f"\nWrote:\n"
        f"  {output_dir / 'features.npy'}\n"
        f"  {output_dir / 'features.csv'}\n"
        f"  {output_dir / 'labels.json'}\n"
        f"  {output_dir / 'category_options.py'}"
    )

    return X, human_labels, category_options


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python format_groundtruth.py <path_to_groundtruth.txt> [output_dir]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "formatted_groundtruth"
    format_groundtruth(in_path, out_dir)