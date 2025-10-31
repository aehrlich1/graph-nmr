from aligner.aligner_AF import process_files
json_folder = '/Users/florianwolf/Desktop/GraphNMR/Alphafold_structures/cleaned_H_added/cleaned_H_distance/'
pdb_folder = "/Users/florianwolf/Desktop/GraphNMR/pdb_structures/first_models"
bmrb_file = "/Users/florianwolf/Desktop/GraphNMR/bmrb_with_pdb.txt"
star_folder = '/Users/florianwolf/Desktop/GraphNMR/bmrb_shifts/'
output_folder = "/Users/florianwolf/Desktop/GraphNMR/aligned_graphs_AF_full"
failed_folder = "/Users/florianwolf/Desktop/GraphNMR/failed_graphs_A_fullF"
alphafold_folder = "/Users/florianwolf/Desktop/GraphNMR/Alphafold_structures/cleaned_H_added"

process_files(json_folder, pdb_folder, star_folder, bmrb_file, output_folder, failed_folder=failed_folder, alphafold_folder=alphafold_folder)
