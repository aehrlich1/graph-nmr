from aligner.aligner import process_files, align_and_map
import pandas as pd
json_folder = '/Users/florianwolf/Desktop/GraphNMR/pdb_structures/distances_test/'
pdb_folder = "/Users/florianwolf/Desktop/GraphNMR/pdb_structures/first_models"
bmrb_file = "/Users/florianwolf/Desktop/GraphNMR/bmrb_with_pdb.txt"
star_folder = '/Users/florianwolf/Desktop/GraphNMR/bmrb_shifts/'
def test_process_files() -> None:
    process_files(json_folder, pdb_folder, star_folder, bmrb_file)


bmrb_df = pd.DataFrame({
    "Residue": list("MTEYKLVV"),
    "Position": range(1, 9)
})

pdb_df = pd.DataFrame({
    "Residue": list("MTEYKLVVVV"),
    "Position": range(10, 20)
})


def test_align_and_map() -> None:
    align_and_map(bmrb_df, pdb_df)
