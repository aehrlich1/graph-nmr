#! /usr/bin/env python3
import sys
from mollib.moleculeNew3 import *
import copy
import logging

def calc_distances(base_dir: str, pdb=True, distance_limit = 6.0, alphafold=False, h_built=False):
    """
    Calculate H-H distances from CIF or PDB files  in `base_dir`
    and save them to JSON in a subfolder.

    Only supports AF3 .cif files

    :param base_dir: directory
    :param pdb: whether to use PDB files or CIF files
    :param distance_limit: upper distance limit for H-H calculation
    :param alphafold: whether the file is an AF prediction
    """


    if pdb and alphafold:
        raise ValueError("pdb and alphafold cannot be true at the same time when calculating H-H distances")

    if alphafold:
        reader = CIFreader_Alphafold
        endswith = ".cif"
        dist_dir = "distances_json_alphafold"
    if h_built:
        reader = CIFreader_Alphafold
        endswith = ".cif"
        dist_dir = "distances_json_h_built"

    elif pdb:
        reader = PDBreader
        endswith = ".pdb"
        dist_dir = "distances_json"
    else:
        reader = CIFreader
        endswith = ".cif"
        dist_dir = "distances_json"

    dist_dir = os.path.join(base_dir, dist_dir)
    os.makedirs(dist_dir, exist_ok=True)



    for filename in os.listdir(base_dir):
        if filename.endswith(endswith):
            try:
                pdb_path = os.path.join(base_dir, filename)
                fnNoPdb = pdb_path[:-4]
                print(fnNoPdb)
                with open(pdb_path, "r") as coordinateFH:
                    frame_ = Frame(
                        reader,
                        PDBWriter,
                        filein=coordinateFH,
                        moleculeMaker=MoleculeMaker_DistanceBased3D,
                    )
                    frame_.read_frame()
                    listOfH = frame_.give_list_of_atom_numbers_of_element("h")

                    distancesToJson, _ = frame_.get_distances_pdb_hydrogens(
                        spaceDistanceTHR=distance_limit,
                        onlyThese=listOfH,
                    )
                    save_state(fnNoPdb + ".json", distancesToJson)

                logging.info(f"Processed {filename}")

            except Exception as e:
                logging.exception(f"Error processing {filename}: {e}")
                print(f"Error processing {filename}: {e}")

    logging.info("Distance calculation completed.")
    print("Done.")


base_dir = "/Users/florianwolf/Desktop/X-ray/Dataset2/x-ray/h_added"
#base_dir = "/Users/florianwolf/Desktop/GraphNMR/Alphafold_structures/cleaned_H_added/"



calc_distances(base_dir, alphafold=False, pdb=False, h_built=True)


