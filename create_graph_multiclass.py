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
import position_encoding_feats


class_of_psp = {
    "baseplate":[1,0,0,0,0,0,0],
    "major_capsid":[0,1,0,0,0,0,0],
    "major_tail":[0,0,1,0,0,0,0],
    "minor_capsid":[0,0,0,1,0,0,0],
    "minor_tail":[0,0,0,0,1,0,0],
    "portal":[0,0,0,0,0,1,0],
    "tail_fiber":[0,0,0,0,0,0,1]
}

graph_data = [ ] # graph_data = [ (node_feature, edge_index, class), (node_feature, edge_index, class), ... , (node_feature, edge_index, class) ] where each tuple represents a protein (its graph and associated node features)


# node feature
def position_encoding(G: nx.Graph) -> nx.Graph:

    for n, d in G.nodes(data=True):
        this_residue_pos = int(d['residue_number']) - 1
        this_residue_pos_encoding = position_encoding_feats.position_encoding_for_individual_residue(this_residue_pos)
        d['pos_encoding'] = this_residue_pos_encoding

    return G

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# node feature, esm per residue embedding 1*L*33 or L*33
def esm_per_residue_embedding(G: nx.Graph, esm_embd_file_path, prot_description) -> nx.Graph:
    
    this_prot_esm_per_residue_embd = np.load(esm_embd_file_path)
    
    for n, d in G.nodes(data=True):
        this_residue_pos = int(d['residue_number']) - 1
        d['esm_embd'] = sigmoid(this_prot_esm_per_residue_embd[0][this_residue_pos])
        
    return G


def create_final_node_features(G: nx.Graph) -> nx.Graph:
    
    for n, d in G.nodes(data=True):
        
        #d['final_node_features'] = np.concatenate((d['amino_acid_one_hot'], d['pos_encoding'], d['esm_embd']))
        d['final_node_features'] = d['amino_acid_one_hot']
        #d['final_node_features'] = d['esm_embd']
        d['final_node_features'] = d['final_node_features'].astype(np.float32)
    
    return G


params_to_change = {
    "granularity": "centroids",
    "node_metadata_functions": [amino_acid_one_hot],
    "edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=1, threshold=10.0)],
    #"graph_metadata_functions": [position_encoding]
    }

def create_graph_and_features_save_to_disk(pdb_file_path, prot_description, esm_embd_file_path, label):
    
    config = ProteinGraphConfig(**params_to_change)
    #print(config.dict())
    g = construct_graph(config=config, pdb_path=pdb_file_path)
    
    #g = esm_per_residue_embedding(g, esm_embd_file_path, prot_description)
    g = create_final_node_features(g)
    
    this_graph_class = np.array(class_of_psp[label])
        
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
    
    graph_data.append((all_node_features, edge_index, this_graph_class))
    
    return 


def parse_metadata(csv_file):
    
    df = pd.read_csv(csv_file)
    d = {} # key = accession and value = label
    for i in range(len(df)):
        this_accession = df.iloc[i]['accession']
        this_label = df.iloc[i]['label']
        d[this_accession] = this_label
        
    return d


def handle_train_data(train_fasta, prot_accession_maps_label, prots_having_pdb, prots_having_esm_embd):
    
    train_pdb_files_path = "/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/phavip_psp_structures"
    train_esm_per_residue_embd_path = "/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/phavip_psp_per_residue_embeddings"
    
    for sequence in SeqIO.parse(train_fasta, "fasta"):
        this_protein_description = str(sequence.description)
        this_accession = this_protein_description.split(" ")[0]
        this_label = prot_accession_maps_label[this_accession]
        if this_accession in prots_having_pdb and this_accession in prots_having_esm_embd:
            
            this_pdb_file_path = os.path.join(train_pdb_files_path, this_accession+".pdb")
            this_esm_per_residue_emdb_file_path = os.path.join(train_esm_per_residue_embd_path, this_accession+".npy")
            create_graph_and_features_save_to_disk(this_pdb_file_path, this_protein_description, this_esm_per_residue_emdb_file_path, this_label)
        
    return

def handle_test_data(test_fasta, prot_accession_maps_label, prots_having_pdb, prots_having_esm_embd):
    
    test_pdb_files_path = "/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/phavip_psp_structures"
    test_esm_per_residue_embd_path = "/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/phavip_psp_per_residue_embeddings"
    
    for sequence in SeqIO.parse(test_fasta, "fasta"):
        this_protein_description = str(sequence.description)
        this_accession = this_protein_description.split(" ")[0]
        this_label = prot_accession_maps_label[this_accession]
        if this_accession in prots_having_pdb and this_accession in prots_having_esm_embd:
            this_pdb_file_path = os.path.join(test_pdb_files_path, this_accession+".pdb")
            this_esm_per_residue_emdb_file_path = os.path.join(test_esm_per_residue_embd_path, this_accession+".npy")
            create_graph_and_features_save_to_disk(this_pdb_file_path, this_protein_description, this_esm_per_residue_emdb_file_path, this_label)
        
    return


def find_proteins_having_pdb_file(pdb_dir):
    pdb_files = os.listdir(pdb_dir)
    prots_having_pdb = {}
    for f in pdb_files:
        this_accession = f.split(".pdb")[0]
        prots_having_pdb[this_accession] = 1
    print(len(prots_having_pdb))
    return prots_having_pdb

def find_proteins_having_esm_embedding(embd_dir):
    embd_files = os.listdir(embd_dir)
    prots_having_esm_embd = {}
    for f in embd_files:
        this_accession = f.split(".npy")[0]
        prots_having_esm_embd[this_accession] = 1
    print(len(prots_having_esm_embd))
    return prots_having_esm_embd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_or_test", "--train_or_test", type = str, required = True, help = "Train seqs or Test seqs (required)")
    parser.add_argument("-t", "--treshold", type = int, required = True, help = "Threshold used while splitting (required)")
    args = parser.parse_args()

    which_data = args.train_or_test
    th = args.treshold
    
    metadata_parent_directory = "/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/metadata_dataset_by_similarity"
    
    prots_having_pdb = find_proteins_having_pdb_file("/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/phavip_psp_structures")
    prots_having_esm_embd = find_proteins_having_esm_embedding("/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/phavip_psp_per_residue_embeddings")
    
    if which_data.lower() == "train":
        
        prot_accession_maps_label = parse_metadata(os.path.join(metadata_parent_directory, "train_"+str(th)+".csv"))
        train_fasta = os.path.join("/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/datasets", "train"+str(th)+".fasta")
        handle_train_data(train_fasta, prot_accession_maps_label, prots_having_pdb, prots_having_esm_embd)
        graph_data = np.array(graph_data)
        np.save(os.path.join("/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/graph_data", "train"+str(th)+"_graphData_ohe.npy"), graph_data)
        
        
    elif which_data.lower() == "test":
        
        prot_accession_maps_label = parse_metadata(os.path.join(metadata_parent_directory, "test_"+str(th)+".csv"))
        test_fasta = os.path.join("/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/datasets", "test"+str(th)+".fasta")
        handle_test_data(test_fasta, prot_accession_maps_label, prots_having_pdb, prots_having_esm_embd)
        graph_data = np.array(graph_data)
        np.save(os.path.join("/home/muhitemon/DeePSP_GIN_on_phavip_split_dataset/graph_data", "test"+str(th)+"_graphData_ohe.npy"), graph_data)
        
    