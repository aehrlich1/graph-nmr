#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 12:04:43 2025

@author: florianwolf
"""

import os
import requests

# === Configuration ===
input_file = "/Users/florianwolf/Desktop/GraphNMR/pdb_ids/pdb_ids_from_bmrbs.txt"       # Your input file with PDB IDs (one per line)
output_dir = "../../../../Desktop/GraphNMR/pdb_fastas"  # Where to save the FASTA files
rcsb_url_template = "https://www.rcsb.org/fasta/entry/{pdb_id}/display"

# === Prepare output directory ===
os.makedirs(output_dir, exist_ok=True)

# === Read and parse comma-separated PDB IDs ===
with open(input_file, "r") as f:
    content = f.read()
    pdb_ids = [pdb_id.strip().upper() for pdb_id in content.split(",") if pdb_id.strip()]

# === Download FASTA for each PDB ID ===
for pdb_id in pdb_ids:
    url = rcsb_url_template.format(pdb_id=pdb_id)
    output_file = os.path.join(output_dir, f"{pdb_id}.fasta")

    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(output_file, "w") as f_out:
            f_out.write(response.text)

        print(f"✅ Downloaded: {pdb_id}")

    except requests.RequestException as e:
        print(f"❌ Failed to download {pdb_id}: {e}")