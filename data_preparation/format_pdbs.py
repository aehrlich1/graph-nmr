# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:14:20 2025

@author: C9Wol
"""

import os
import gzip
import shutil

# Define the directory containing your zip files
directory = "/Users/florianwolf/Desktop/GraphNMR/pdb_structures_cif"




for filename in os.listdir(directory):
    if filename.endswith(".pdb"):  # Ensure only PDB files are processed
        pdb_path = os.path.join(directory, filename)
        output_path = os.path.join(directory, f"first_model_{filename}")

        with open(pdb_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                outfile.write(line)
                if line.startswith("ENDMDL"):  # Indicates the end of the first model
                    break

print("Extraction complete!")


for filename in os.listdir(directory):
    if filename.endswith(".pdb"):  # Ensure only PDB files are processed
        pdb_path = os.path.join(directory, filename)
        output_path = os.path.join(directory, f"m1_{filename}")

        with open(pdb_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                outfile.write(line)
                if line.startswith("ENDMDL"):  # Indicates the end of the first model
                    break

print("First model complete!")


for filename in os.listdir(directory):
    if filename.endswith(".pdb"):
        pdb_path = os.path.join(directory, filename)
        output_path = os.path.join(directory, f"mod_{filename}")

        with open(pdb_path, 'r') as infile, open(output_path, 'w') as outfile:
            first_residue = None  # Track the first residue
            for line in infile:
                if line.startswith("ATOM"):
                    residue_id = line[22:26].strip()  # Extract residue sequence number
                    
                    if first_residue is None:
                        first_residue = residue_id  # Identify the first residue

                    if residue_id == first_residue:
                        if "H1" in line or "H2" in line:
                            continue
                        if "H3" in line:
                            line = line[:12] + " H  " + line[16:]
                        

                    if (" OXT" in line):
                        continue  # Skip C-terminal oxygen and N-terminal hydrogens
                        
                    if ("HE2 HIS" in line):
                        continue

                outfile.write(line)

print("Modification complete!")


