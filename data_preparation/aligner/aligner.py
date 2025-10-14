from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1
import pandas as pd
import os
import glob
import logging
import pynmrstar
import networkx as nx

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



def process_files(json_folder, pdb_folder, star_folder, bmrb_file, output_folder, threshold=1.3, failed_folder=None, verbose=False) -> None:
    logger = setup_logger(verbose=verbose)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(failed_folder, exist_ok=True)
    for json_file in glob.glob(os.path.join(json_folder, "*.json")):
        try:
            # prepare pdb
            pdb_id = os.path.basename(json_file).split('_')[1].split('.')[0]
            pdb_file_path = os.path.join(pdb_folder, f"m1_{pdb_id}.pdb")
            parser = PDBParser()
            structure = parser.get_structure("prot", pdb_file_path)[0]

            # prepare bmrb-pdb correspondance
            bmrb_with_pdb_id = pd.read_csv(bmrb_file, sep=",", names=["bmrb_id", "pdb_id"])
            bmrb_id_row = bmrb_with_pdb_id[bmrb_with_pdb_id["pdb_id"] == pdb_id]
            if bmrb_id_row.empty:
                logger.warning(f"No BMRB ID found for PDB ID: {pdb_id}. Skipping...")
                continue
            bmrb_id = bmrb_id_row["bmrb_id"].values[0]

            # get bmrb star file with chemical shifts
            star_file_path = os.path.join(star_folder, f"bmr{bmrb_id}", f"bmr{bmrb_id}_3.str")
            if not os.path.exists(star_file_path):
                logger.warning(f"STAR file not found for BMRB ID: {bmrb_id}. Skipping...")
                continue
            star_file = pynmrstar.Entry.from_file(star_file_path)

            loop = star_file.get_loops_by_category("Atom_chem_shift")[0]
            bmrb_columns = ['Comp_index_ID', 'Comp_ID', 'Atom_ID', 'Atom_type', 'Val', 'Val_err']
            cs_data = pd.DataFrame(loop.get_tag(bmrb_columns), columns=bmrb_columns)


            # three letter sequence
            pdb_sequence = get_pdb_sequence(structure)
            bmrb_sequence = get_bmrb_sequence(cs_data)

            # Do alignment
            best_alignment, mapping = align_and_map(bmrb_sequence, pdb_sequence)

            # Get cs data pdb atom name -> bmrb_name
            mapping_reset = mapping.reset_index()  # turns pdb_residue/pdb_position into columns

            merged = cs_data.merge(
                mapping_reset,
                left_on=["Comp_ID", "Comp_index_ID"],
                right_on=["bmrb_residue", "bmrb_position"],
                how="left"
            )
            # Do Graph, everything is in the merged table already #FIXME pdb_label is not float
            merged["label"] = merged["Atom_ID"] + " " + merged["pdb_residue"] + " " + merged["pdb_position"].astype("Int64").astype(str)

            json_df = pd.read_json(json_file).T
            num_residues = json_df["aminoacidNum0"].astype(int).max()
            json_df["label_0"] = json_df[["pdbAtomName0", "aminoacid0", "aminoacidNum0"]].astype(str).agg(" ".join, axis=1)
            json_df["label_1"] = json_df[["pdbAtomName1", "aminoacid1", "aminoacidNum1"]].astype(str).agg(" ".join, axis=1)
            json_df = json_df.drop(columns=["pdbAtomName0", "aminoacid0", "aminoacidNum0", "pdbAtomName1", "aminoacid1", "aminoacidNum1"])


            G = nx.Graph()

            # Add nodes and edges
            for _, row in json_df.iterrows():
                try:
                    node1_name = f"{row['label_0']}"
                    node2_name = f"{row['label_1']}"

                    cs_node1 = merged[merged["label"] == node1_name]["Val"].iloc[0]
                    cs_node2 = merged[merged["label"] == node2_name]["Val"].iloc[0]
                    cs_node1_err = merged[merged["label"] == node1_name]["Val_err"].iloc[0]
                    cs_node2_err = merged[merged["label"] == node2_name]["Val_err"].iloc[0]

                    distance = row['d']

                    G.add_node(node1_name, chem_shift=cs_node1, chem_shift_error=cs_node1_err)
                    G.add_node(node2_name, chem_shift=cs_node2, chem_shift_error=cs_node2_err)
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

def get_bmrb_sequence(cs_data) -> pd.DataFrame:
    sequence = (
        cs_data[["Comp_ID", "Comp_index_ID"]]
        .drop_duplicates()
        .rename(columns={"Comp_ID": "Residue", "Comp_index_ID": "Position"})
    )

    return sequence


def align_and_map(bmrb_sequence: pd.DataFrame, pdb_sequence: pd.DataFrame):
    """
    Aligns BMRB and PDB sequences, returns alignment + mapping table.

    bmrb_sequence, pdb_sequence: DataFrames with at least
        - 'residue': one-letter code
        - 'index': residue index (e.g., Comp_index_ID from BMRB, resseq from PDB)
    """

    # Convert to strings for alignment
    bmrb_seq_str = "".join(bmrb_sequence["Residue"])
    pdb_seq_str = "".join(pdb_sequence["Residue"])

    # Set up aligner #NOTE alignment is done in 1 letter code, but it shall be fine given the relative numbering would match
    aligner = PairwiseAligner()
    aligner.mode = "global"  # full-length alignment
    alignments = aligner.align(seq1(bmrb_seq_str), seq1(pdb_seq_str))
    best = alignments[0]

    # Extract alignment blocks
    bmrb_blocks, pdb_blocks = best.aligned  # list of (start, end) tuples

    mapping = []
    for (b_start, b_end), (p_start, p_end) in zip(bmrb_blocks, pdb_blocks):
        for b_idx, p_idx in zip(range(b_start, b_end), range(p_start, p_end)):
            mapping.append({
                "bmrb_residue": bmrb_sequence.iloc[b_idx]["Residue"],
                "bmrb_position": bmrb_sequence.iloc[b_idx]["Position"],
                "pdb_residue": pdb_sequence.iloc[p_idx]["Residue"],
                "pdb_position": pdb_sequence.iloc[p_idx]["Position"],
            })

    mapping_df = pd.DataFrame(mapping).set_index(["pdb_residue", "pdb_position"])
    return best, mapping_df



# Lookup with pdb naming
def pdb_to_bmrb(df, pdb_res, pdb_pos):
    try:
        row = df.loc[(pdb_res, pdb_pos)]
        return row["bmrb_residue"], row["bmrb_position"]
    except KeyError:
        return None  # no match (gap case)
