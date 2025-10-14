from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1, seq3
import pandas as pd
import os
import glob
import logging
import pynmrstar
import networkx as nx
from Bio.PDB import MMCIFParser, Polypeptide

def setup_logger(verbose):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        filename='graph_from_json.log',
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    return logger



def process_files(json_folder, pdb_folder, star_folder, bmrb_file, output_folder, threshold=1.3, alphafold_folder=None,failed_folder=None) -> None:
    logger = setup_logger(verbose=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(failed_folder, exist_ok=True)
    for json_file in glob.glob(os.path.join(json_folder, "*.json")):
        try:
            # prepare pdb
            pdb_id = os.path.basename(json_file).split('_')[0].upper() #TODO remove the fold aswell, make em upper
            pdb_file_path = os.path.join(pdb_folder, f"m1_{pdb_id}.pdb")
            parser = PDBParser()
            pdb_structure = parser.get_structure("prot", pdb_file_path)[0]

            # get bmrb star file with chemical shifts
            star_file_path = os.path.join(alphafold_folder, f"{pdb_id.lower()}_modelH.cif")
            if not os.path.exists(star_file_path):
                logger.warning(f"STAR file not found for BMRB ID: {pdb_id}. Skipping...")
                continue
            parser = MMCIFParser(QUIET=True)
            fold_structure = parser.get_structure("protein", star_file_path)

            ppb = PPBuilder()
            for model in fold_structure:
                for chain in model:
                    peptides = ppb.build_peptides(chain)
                    if peptides:
                        for pp in peptides:
                            if chain.id == "A":
                                fold_sequence = pp.get_sequence()


            # three letter sequence
            pdb_sequence = get_pdb_sequence(pdb_structure)


            # Do alignment
            best_alignment, mapping = align_and_map(fold_sequence, pdb_sequence)


            #
            mapping_reset = mapping.reset_index()  # turns pdb_residue/pdb_position into columns
            mapping_reset["pdb"] = mapping_reset[["pdb_residue", "pdb_position"]].astype(str).agg(" ".join, axis=1)
            mapping_reset["alphafold"] = mapping_reset[["alphafold_residue", "alphafold_position"]].astype(str).agg(" ".join, axis=1)




            json_df = pd.read_json(json_file).T
            num_residues = json_df["aminoacidNum0"].astype(int).max()
            json_df["label_0"] = json_df[["aminoacid0", "aminoacidNum0"]].astype(str).agg(" ".join, axis=1)
            json_df["label_1"] = json_df[["aminoacid1", "aminoacidNum1"]].astype(str).agg(" ".join, axis=1)
            #json_df = json_df.drop(columns=["pdbAtomName0", "aminoacid0", "aminoacidNum0", "pdbAtomName1", "aminoacid1", "aminoacidNum1"])

            # Create lookup dictionary from alphafold → pdb
            af_to_pdb = mapping_reset.set_index("alphafold")["pdb"].to_dict()

            # Map each AlphaFold label to the corresponding PDB label
            json_df["pdb_match_0"] = json_df["label_0"].map(af_to_pdb)
            json_df["pdb_match_1"] = json_df["label_1"].map(af_to_pdb)

            # Combine pdbAtomName with the mapped pdb residue info
            json_df["pdb_label_0"] = (
                    json_df["pdbAtomName0"].astype(str) + " " + json_df["pdb_match_0"].astype(str)
            )
            json_df["pdb_label_1"] = (
                    json_df["pdbAtomName1"].astype(str) + " " + json_df["pdb_match_1"].astype(str)
            )


            G = nx.Graph()

            # Add nodes and edges
            for _, row in json_df.iterrows():
                try:
                    node1_name = row["pdb_label_0"]
                    node2_name = row["pdb_label_1"]
                    if "nan" in node1_name:
                        continue
                    if "nan" in node2_name:
                        continue
                    node1_plDDT = row["plDDT0"]
                    node2_plDDT = row["plDDT1"]


                    distance = row['d']

                    G.add_node(node1_name, plDDT=node1_plDDT)
                    G.add_node(node2_name, plDDT=node2_plDDT)
                    G.add_edge(node1_name, node2_name, weight=distance, label=f"{distance:.2f}")


                except (IndexError, KeyError) as e:
                    logger.debug(f"Skipping row due to missing data: {e}")
                    logger.debug(f"Atom name {node1_name} and {node2_name}")
                    continue

            num_nodes = len(G.nodes)
            if num_nodes >= threshold * num_residues:
                output_graph_path = os.path.join(output_folder, f"{pdb_id}_graph.gml")
                nx.write_gml(G, output_graph_path)
                logger.debug(f"Graph saved for PDB ID {pdb_id} at {output_graph_path}")
            else:
                failed_graph_path = os.path.join(failed_folder, f"{pdb_id}_graph.gml")
                nx.write_gml(G, failed_graph_path)
                logger.debug(f"Graph saved for PDB ID {pdb_id} at {failed_graph_path}")
                logger.warning(f"Graph for PDB ID {pdb_id} skipped: only {num_nodes} nodes, threshold was {1.3 * num_residues:.1f}")




        except Exception as e:
            logger.error(f"An error occurred while processing {json_file}: {e}")
    logger.info("Done")
def get_pdb_sequence(structure) -> pd.DataFrame:
    pdb_residues = []
    # chain A
    for res in structure["A"]:
        hetflag = res.id[0] # non-standard amino acids will be omitted
        if hetflag != " ":
            continue
        resname = res.get_resname()
        resnum = res.id[1]
        pdb_residues.append([resname, resnum])

    sequence = pd.DataFrame(pdb_residues, columns=["Residue", "Position"])
    return sequence

def get_alphafold_sequence(cs_data) -> pd.DataFrame:
    sequence = (
        cs_data[["Comp_ID", "Comp_index_ID"]]
        .drop_duplicates()
        .rename(columns={"Comp_ID": "Residue", "Comp_index_ID": "Position"})
    )

    return sequence


def align_and_map(alphafold_sequence: pd.DataFrame, pdb_sequence: pd.DataFrame):
    """
    Aligns BMRB and PDB sequences, returns alignment + mapping table.

    bmrb_sequence, pdb_sequence: DataFrames with at least
        - 'residue': one-letter code
        - 'index': residue index (e.g., Comp_index_ID from BMRB, resseq from PDB)
    """

    # Convert to strings for alignment
    alphafold_seq_str = "".join(alphafold_sequence) # TODO change for fold
    pdb_seq_str = "".join(pdb_sequence["Residue"]) #

    # Set up aligner #NOTE alignment is done in 1 letter code, but it shall be fine given the relative numbering would match
    aligner = PairwiseAligner()
    aligner.mode = "global"  # full-length alignment
    alignments = aligner.align(alphafold_seq_str, seq1(pdb_seq_str))
    best = alignments[0]

    # Extract alignment blocks
    alphafold_blocks, pdb_blocks = best.aligned  # list of (start, end) tuples

    mapping = []
    for (b_start, b_end), (p_start, p_end) in zip(alphafold_blocks, pdb_blocks):
        for b_idx, p_idx in zip(range(b_start, b_end), range(p_start, p_end)):
            mapping.append({
                "alphafold_residue": seq3(alphafold_sequence[b_idx]).upper(),
                "alphafold_position": b_idx+1,
                "pdb_residue": pdb_sequence.iloc[p_idx]["Residue"],
                "pdb_position": pdb_sequence.iloc[p_idx]["Position"],
            })

    #TODO convert back to three letter in bmrb/fold case
    mapping_df = pd.DataFrame(mapping).set_index(["pdb_residue", "pdb_position"])
    return best, mapping_df



# Lookup with pdb naming
def pdb_to_bmrb(df, pdb_res, pdb_pos):
    try:
        row = df.loc[(pdb_res, pdb_pos)]
        return row["alphafold_residue"], row["alphafold_position"]
    except KeyError:
        return None  # no match (gap case)
