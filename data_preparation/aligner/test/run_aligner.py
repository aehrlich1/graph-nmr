from aligner.aligner import process_files
json_folder = '/Users/florianwolf/Desktop/GraphNMR/pdb_structures/distances_json/'
pdb_folder = "/Users/florianwolf/Desktop/GraphNMR/pdb_structures/first_models"
bmrb_file = "/Users/florianwolf/Desktop/GraphNMR/bmrb_with_pdb.txt"
star_folder = '/Users/florianwolf/Desktop/GraphNMR/bmrb_shifts/'
output_folder = "/Users/florianwolf/Desktop/GraphNMR/aligned_graphs_NMR"
failed_folder = "/Users/florianwolf/Desktop/GraphNMR/failed_graphs_NMR"

process_files(json_folder, pdb_folder, star_folder, bmrb_file, output_folder, failed_folder=failed_folder)
