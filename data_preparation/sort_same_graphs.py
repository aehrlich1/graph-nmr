import os
import shutil
import logging

# --- CONFIGURE THESE PATHS ---
folder1 = "/Users/florianwolf/Desktop/GraphNMR/monomer_graphs_experimental"
folder2 = "/Users/florianwolf/Desktop/GraphNMR/aligned_graphs_AF_full/" # all AF graphs but should work
output_root = "NMR_graphs"
log_file = "../../../Desktop/GraphNMR/sort_same_graphs.log"

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

# Make sure output folder exists
os.makedirs(output_root, exist_ok=True)

# Helper: get pdb id (first 4 chars, lowercase)
def get_pdb_id(filename):
    return filename[:4].lower()

# Map pdb_id -> filename
files1 = {get_pdb_id(f): f for f in os.listdir(folder1)}
files2 = {get_pdb_id(f): f for f in os.listdir(folder2)}

# Find common pdb IDs
common_ids = set(files1.keys()) & set(files2.keys())

for pdb_id in common_ids:
    target_dir = os.path.join(output_root, pdb_id.upper())
    os.makedirs(target_dir, exist_ok=True)

    shutil.copy(os.path.join(folder1, files1[pdb_id]), target_dir)
    shutil.copy(os.path.join(folder2, files2[pdb_id]), target_dir)

    logging.info(f"Matched PDB ID {pdb_id.upper()}: copied {files1[pdb_id]} and {files2[pdb_id]}")

# Report unmatched files
unmatched1 = set(files1.keys()) - common_ids
unmatched2 = set(files2.keys()) - common_ids

if unmatched1:
    logging.warning("No matches for files in folder1:")
    for pdb_id in unmatched1:
        logging.warning(f"  {files1[pdb_id]}")

if unmatched2:
    logging.warning("No matches for files in folder2:")
    for pdb_id in unmatched2:
        logging.warning(f"  {files2[pdb_id]}")

logging.info(f"Done! Stored {len(common_ids)} matching PDB sets in '{output_root}'.")
logging.info(f"Full log written to {log_file}")     
