from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
#from graphein.protein.visualisation import plotly_protein_structure_graph
from functools import partial
from graphein.protein.edges.distance import add_distance_threshold
import networkx as nx
import numpy as np
import json
import os
import sys
from numpy.linalg import norm
import position_encoding_feats


class_of_psp = {
    "baseplate":[1,0,0,0,0,0,0,0,0,0,0],
    "collar":[0,1,0,0,0,0,0,0,0,0,0],
    "HTJ":[0,0,1,0,0,0,0,0,0,0,0],
    "major_capsid":[0,0,0,1,0,0,0,0,0,0,0],
    "major_tail":[0,0,0,0,1,0,0,0,0,0,0],
    "minor_capsid":[0,0,0,0,0,1,0,0,0,0,0],
    "minor_tail":[0,0,0,0,0,0,1,0,0,0,0],
    "other":[0,0,0,0,0,0,0,1,0,0,0],
    "portal":[0,0,0,0,0,0,0,0,1,0,0],
    "shaft":[0,0,0,0,0,0,0,0,0,1,0],
    "tail_fiber":[0,0,0,0,0,0,0,0,0,0,1]
}

avg_mass_of_aa = {

    "GLY":57.05132, "ALA":71.0779, "SER":87.0773, "PRO":97.11518, "VAL":99.13106, "THR":101.10388, "CYS":103.1429, "LEU":113.15764, "ILE":113.15764, "ASN":114.10264, "ASP":115.0874, "GLN":128.12922,
    "LYS":128.17228, "GLU":129.11398, "MET":131.19606, "HIS":137.13928, "PHE":147.17386, "SEC":150.3079, "ARG":156.18568, "TYR":163.17326, "TRP":186.2099, "PYL":237.29816
}


graph_data_each_bin = [ ] # graphs_of_XD = [ (node_feature, edge_index, edge_attr, class), (node_feature, edge_index, edge_attr, class), ... , (node_feature, edge_index, edge_attr, class) ]

'''
def find_COM(aa_with_coords):
    """
    input-> [(aa,[x,y,z]), (aa,[x,y,z])]
    """
    sum_of_mass = 0
    for (aa, coords) in aa_with_coords:
        sum_of_mass+=avg_mass_of_aa[aa]
    
    COM_x, COM_y, COM_z = 0, 0, 0

    for (aa, coords) in aa_with_coords:
        COM_x+=avg_mass_of_aa[aa]*coords[0]
        COM_y+=avg_mass_of_aa[aa]*coords[1]
        COM_z+=avg_mass_of_aa[aa]*coords[2]
    
    COM_x = COM_x / sum_of_mass
    COM_y = COM_y / sum_of_mass
    COM_z = COM_z / sum_of_mass
    COM = np.array([COM_x, COM_y, COM_z])
    
    return COM

# edge feature
def orientation_between_two_amino_acids(G: nx.Graph) -> nx.Graph:

    residue_number_maps_coords = {}
    aa_coords_list = []
    for n, d in G.nodes(data=True):
        aa_name = str(n.split(':')[1])
        aa_coords_list.append((aa_name, d['coords']))
        this_residue_number = int(d['residue_number'])
        residue_number_maps_coords[this_residue_number] = d['coords']

    #COM = np.average(coords_list, axis=0)
    COM = find_COM(aa_coords_list)

    for u, v, d in G.edges(data=True):

        u_residue_number = u.split(':')[-1]
        u_residue_number = int(u_residue_number)

        v_residue_number = v.split(':')[-1]
        v_residue_number = int(v_residue_number)

        vector_A = np.array(residue_number_maps_coords[u_residue_number]) - COM
        vector_B = np.array(residue_number_maps_coords[v_residue_number]) - COM

        orientation_between_vector_A_and_B = np.dot(vector_A, vector_B)/(norm(vector_A)*norm(vector_B))

        d['orientation'] = orientation_between_vector_A_and_B

    return G
'''

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
def esm_per_residue_embedding(G: nx.Graph, bin_number_with_class, prot_description) -> nx.Graph:
    
    parent_path_esm_embd_dir = "/home/muhitemon/esm_gnn_project_per_residue_embeddings"
    this_protein_esm_per_residue_embd_file = parent_path_esm_embd_dir + "/" + bin_number_with_class + "/" + prot_description + ".npy"
    this_prot_esm_per_residue_embd = np.load(this_protein_esm_per_residue_embd_file)
    
    for n, d in G.nodes(data=True):
        this_residue_pos = int(d['residue_number']) - 1
        d['esm_embd'] = sigmoid(this_prot_esm_per_residue_embd[0][this_residue_pos])
    return G


params_to_change = {
    "granularity": "centroids",
    "node_metadata_functions": [amino_acid_one_hot],
    "edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=1, threshold=10.)],
    "graph_metadata_functions": [position_encoding]
    }


def create_final_node_and_edge_features(G: nx.Graph) -> nx.Graph:
    
    for n, d in G.nodes(data=True):
        #d['final_node_features'] = np.concatenate((d['amino_acid_one_hot'], d['pos_encoding'], d['esm_embd'], np.array([0,0])))
        d['final_node_features'] = np.concatenate((d['amino_acid_one_hot'], d['pos_encoding'], d['esm_embd']))
        d['final_node_features'] = d['final_node_features'].astype(np.float32)
    
    '''
    prefix_zeros = np.zeros(73)
    for u, v, d in G.edges(data=True):
        dis = np.array([float(d['distance'])])
        ornt = np.array([float(d['orientation'])])
        d['final_edge_features'] = np.concatenate((prefix_zeros, dis, ornt))
        d['final_edge_features'] = d['final_edge_features'].astype(np.float32)
    '''
    
    return G


def create_graph_and_features_save_to_disk(pdb_file_path):
    
    config = ProteinGraphConfig(**params_to_change)
    #print(config.dict())
    g = construct_graph(config=config, pdb_path=pdb_file_path)
    bin_number_with_class, pdb_file_name = pdb_file_path.split('/')[-2], pdb_file_path.split('/')[-1]
    prot_description = pdb_file_name.split(".pdb")[0]
    
    g = esm_per_residue_embedding(g, bin_number_with_class, prot_description)
    g = create_final_node_and_edge_features(g)
    
    tmp = bin_number_with_class.split('_')
    bin_number = int(tmp[0])
    if len(tmp) == 2:
        class_name = tmp[1]
    elif len(tmp) == 3:
        class_name = tmp[1] + "_" + tmp[2]
    
    all_node_features = []
    this_graph_class = np.array(class_of_psp[class_name])
    for n, d in g.nodes(data=True):
        all_node_features.append(d['final_node_features'])
    
    num_nodes = len(all_node_features)-1
    all_node_features = np.array(all_node_features).astype(np.float32)
    
    edge_index = [ [],[] ]
    #all_edge_attrs = []
    
    flag = 1
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
            
            #all_edge_attrs.append(d['final_edge_features'])
            #all_edge_attrs.append(d['final_edge_features'])
        
        else:
            flag = 0
    
    if flag == 0:
        print("{} >>> {}".format(bin_number_with_class, pdb_file_name))
        
    #all_edge_attrs = np.array(all_edge_attrs).astype(np.float32)
    edge_index = np.array(edge_index)
    
    '''
    phanns_feats_parent_dir = "/home/muhitemon/phanns_paper_features"
    this_protein_phanns_feat_file = phanns_feats_parent_dir + "/" + bin_number_with_class + "/" + prot_description + ".npy"
    this_prot_phanns_feats = np.load(this_protein_phanns_feat_file)
    this_prot_phanns_feats = this_prot_phanns_feats.astype(np.float32)
    '''
    
    graph_data_each_bin.append((all_node_features, edge_index, this_graph_class))
    
    return 




if __name__ == '__main__':
    
    which_bin = str(sys.argv[1])
    
    # load the saved JSON file of proteins having esm embeddings back into a dictionary
    with open("/home/muhitemon/prots_having_esm_embd.json", "r") as f:
        prots_with_esm_embd = json.load(f) # {"1_portal":{protein_description:1}, "2_HJT":{protein_description:1}}
    
    parent_path_of_structures_dir = "/home/muhitemon/esm_predicted_structures_way2"
    classes = os.listdir(parent_path_of_structures_dir)
    
    
    for i in classes:
        this_class_name = i
        tmp = this_class_name.split('_')
        bin_number = str(tmp[0])
        
        if bin_number != which_bin:
            continue
        
        this_class_pdb_path = os.path.join(parent_path_of_structures_dir, this_class_name)
        
        this_class_prot_pdb_files = os.listdir(this_class_pdb_path)
        
        
        for pdb_file in this_class_prot_pdb_files:
            this_protein_description = pdb_file.split(".pdb")[0]
            if this_protein_description in prots_with_esm_embd[this_class_name]:
                this_pdb_file_path = os.path.join(this_class_pdb_path, pdb_file)
                create_graph_and_features_save_to_disk(this_pdb_file_path)
                
                
        
        print(this_class_name)
    #graph_data_bin_wise_dir = "/home/muhitemon/graph_data_bin_wise"
    
    graph_data_each_bin = np.array(graph_data_each_bin)
    
    if which_bin == "1":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/1D/1_graphData.npy", graph_data_each_bin)
        
    elif which_bin == "2":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/2D/2_graphData.npy", graph_data_each_bin)
        
    elif which_bin == "3":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/3D/3_graphData.npy", graph_data_each_bin)
        
    elif which_bin == "4":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/4D/4_graphData.npy", graph_data_each_bin)
        
    elif which_bin == "5":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/5D/5_graphData.npy", graph_data_each_bin)
        
    elif which_bin == "6":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/6D/6_graphData.npy", graph_data_each_bin)
    
    elif which_bin == "7":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/7D/7_graphData.npy", graph_data_each_bin)
    
    elif which_bin == "8":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/8D/8_graphData.npy", graph_data_each_bin)
    
    elif which_bin == "9":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/9D/9_graphData.npy", graph_data_each_bin)
    
    elif which_bin == "10":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/10D/10_graphData.npy", graph_data_each_bin)
        
    elif which_bin == "11":
        np.save("/home/muhitemon/graph_data_bin_wise_wo_edge_features/11D/11_graphData.npy", graph_data_each_bin)
    
