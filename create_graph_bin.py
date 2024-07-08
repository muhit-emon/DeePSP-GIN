from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
#from graphein.protein.visualisation import plotly_protein_structure_graph
from functools import partial
from graphein.protein.edges.distance import add_distance_threshold
import networkx as nx
import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
from numpy.linalg import norm
from Bio import SeqIO

graph_data = [ ] # graph_data = [ (node_feature, edge_index), (node_feature, edge_index), ... , (node_feature, edge_index) ] where each tuple represents a protein (its graph and associated node features)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# node feature, esm per residue embedding 1*L*33 or L*33
def esm_per_residue_embedding(G: nx.Graph, esm_embd_file_path) -> nx.Graph:
    
    this_prot_esm_per_residue_embd = np.load(esm_embd_file_path)
    
    for n, d in G.nodes(data=True):
        this_residue_pos = int(d['residue_number']) - 1
        d['esm_embd'] = sigmoid(this_prot_esm_per_residue_embd[0][this_residue_pos])
        
    return G


def create_final_node_features(G: nx.Graph) -> nx.Graph:
    
    for n, d in G.nodes(data=True):
        
        d['final_node_features'] = d['esm_embd']
        d['final_node_features'] = d['final_node_features'].astype(np.float32)
    
    return G

def create_graph_and_features_save_to_disk(pdb_file_path, esm_embd_file_path, params_to_change):
    
    config = ProteinGraphConfig(**params_to_change)
    #print(config.dict())
    g = construct_graph(config=config, pdb_path=pdb_file_path)
    
    g = esm_per_residue_embedding(g, esm_embd_file_path)
    g = create_final_node_features(g)
    
    all_node_features = []
    
    for n, d in g.nodes(data=True):
        all_node_features.append(d['final_node_features'])
    
    num_nodes = len(all_node_features)-1
    all_node_features = np.array(all_node_features).astype(np.float32)
    
    edge_index = [ [],[] ]
    
    for u, v, d in g.edges(data=True):

        u_residue_number = u.split(':')[-1]
        u_residue_number = int(u_residue_number) - 1

        v_residue_number = v.split(':')[-1]
        v_residue_number = int(v_residue_number) - 1
        
        if u_residue_number >= 0 and u_residue_number <= num_nodes and v_residue_number >= 0 and v_residue_number <= num_nodes:
        
            edge_index[0].append(u_residue_number)
            edge_index[1].append(v_residue_number)
            
            edge_index[0].append(v_residue_number)
            edge_index[1].append(u_residue_number)
        
    
    edge_index = np.array(edge_index)
    
    graph_data.append((all_node_features, edge_index))
    
    return

def handle_data(fasta_file, prots_having_pdb, prots_having_esm_embd, params_to_change, esm_embedding_dir, pdb_dir):
    
    d = {}
    idx = 0

    for sequence in SeqIO.parse(fasta_file, "fasta"):
        this_protein_description = str(sequence.description)
        this_prot_ID = this_protein_description.split("|")[0]
        
        if this_prot_ID in prots_having_pdb and this_prot_ID in prots_having_esm_embd:
            
            this_pdb_file_path = os.path.join(pdb_dir, this_prot_ID+".pdb")
            this_esm_per_residue_emdb_file_path = os.path.join(esm_embedding_dir, this_prot_ID+".npy")
            create_graph_and_features_save_to_disk(this_pdb_file_path, this_esm_per_residue_emdb_file_path, params_to_change)
            d[idx] = this_prot_ID
            idx+=1
    
    # Open a file in write mode
    with open('index_maps_ID.json', 'w') as file:
        # Use json.dump() to write the dictionary to the file
        json.dump(d, file)
    
    return


def find_proteins_having_pdb_file(pdb_dir):
    pdb_files = os.listdir(pdb_dir)
    prots_having_pdb = {}
    for f in pdb_files:
        this_accession = f.split(".pdb")[0]
        prots_having_pdb[this_accession] = 1
    #print(len(prots_having_pdb))
    return prots_having_pdb

def find_proteins_having_esm_embedding(embd_dir):
    embd_files = os.listdir(embd_dir)
    prots_having_esm_embd = {}
    for f in embd_files:
        this_accession = f.split(".npy")[0]
        prots_having_esm_embd[this_accession] = 1
    #print(len(prots_having_esm_embd))
    return prots_having_esm_embd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-protein_fasta_with_ID", "--protein_fasta_with_ID", type = str, required = True, help = "Protein Fasta File with Protein IDs (required)")
    parser.add_argument("-esm_embedding_dir", "--esm_embedding_dir", type = str, required = True, help = "ESM embedding directory (required)")
    parser.add_argument("-pdb_dir", "--pdb_dir", type = str, required = True, help = "PDB directory (required)")
    args = parser.parse_args()

    protein_fasta_file_with_ID = args.protein_fasta_with_ID
    esm_embedding_dir = args.esm_embedding_dir
    pdb_dir = args.pdb_dir
    
    th = 10.0
    params_to_change = {
        "granularity": "centroids",
        "edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=1, threshold=th)],
    }
    
    
    proteins_having_pdb = find_proteins_having_pdb_file(pdb_dir)
    proteins_having_esm_embd = find_proteins_having_esm_embedding(esm_embedding_dir)
    handle_data(protein_fasta_file_with_ID, proteins_having_pdb, proteins_having_esm_embd, params_to_change, esm_embedding_dir, pdb_dir)
    
    graph_data = np.array(graph_data)
    print(graph_data.shape)
    np.save("graph.npy", graph_data)
    