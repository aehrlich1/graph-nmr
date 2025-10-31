#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 11:30:52 2025

@author: florianwolf
"""

# This script takes an input file with pdb ids and scans a folder of files if they are in the id file
# then it sorts them into two folders

import os
import shutil

# Pfade anpassen
pdb_list_file = "/Users/florianwolf/Desktop/GraphNMR/pdb_ids/monomers_nmr_pdb_ids.txt"
graphs_folder = "/Users/florianwolf/Desktop/GraphNMR/aligned_graphs_NMR"
monomer_folder = "/Users/florianwolf/Desktop/GraphNMR/monomer_graphs_experimental"
polymer_folder = "/Users/florianwolf/Desktop/GraphNMR/polymer_graphs_experimental"

# Ausgabeordner erstellen, falls nicht vorhanden
os.makedirs(monomer_folder, exist_ok=True)
os.makedirs(polymer_folder, exist_ok=True)

# PDB IDs einlesen (ohne Leerzeichen/Zeilenumbrüche)
with open(pdb_list_file, "r") as f:
    content = f.read().strip().lower()
    pdb_ids = [pdb_id.strip() for pdb_id in content.split(",") if pdb_id.strip()]
    

# Dateien im graphs-Ordner durchgehen
for filename in os.listdir(graphs_folder):
    file_path = os.path.join(graphs_folder, filename)
    
    # Nur Dateien berücksichtigen
    if os.path.isfile(file_path):
        # Prüfen, ob eine PDB ID im Dateinamen vorkommt
        if any(pdb_id in filename.lower() for pdb_id in pdb_ids):
            dest_path = os.path.join(monomer_folder, filename)
        else:
            dest_path = os.path.join(polymer_folder, filename)
        
        # Datei kopieren
        shutil.copy2(file_path, dest_path)
        print(f"{filename} → {dest_path}")

print("Fertig!")
        