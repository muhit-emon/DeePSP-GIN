######################################################################################################################################################
#      This program generates Lx33 feature set from an input fasta and outputs in npy format                                                         #
#      Credit: facebookresearch,                                                                                                                     #
#      colab notebook https://colab.research.google.com/github/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb                    #
######################################################################################################################################################
import esm
import torch
import os
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
import numpy
import argparse
torch.set_grad_enabled(False)
import numpy as np

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

# This function generates Lx33 feature set for an individual protein sequence and outputs in npy format. shape->(1, L, 33)
def generate_esm_per_residue_embedding(protein_description, protein_seq):

    #out_npy = "./test.npy"
    seq_transformer, seq_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    seq_transformer = seq_transformer.eval()#.cuda()
    seq_batch_converter = seq_alphabet.get_batch_converter()
    seq_data = [(protein_description, protein_seq)]
    seq_batch_labels, seq_batch_strs, seq_batch_tokens = seq_batch_converter(seq_data)
    seq_np = seq_batch_tokens.numpy()
    out1 = seq_transformer(seq_batch_tokens)
    out2 = out1['logits'].numpy()
    #np.save(out_npy, out2)
    return out2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-protein_fasta_with_ID", "--protein_fasta_with_ID", type = str, required = True, help = "Protein Fasta File with Protein IDs (required)")
    args = parser.parse_args()

    protein_fasta_file_with_ID = args.protein_fasta_with_ID

    
    # create esm embedding directory inside nf work directory
    embedding_dir = 'esm_embedding_dir'
    # Create the directory
    os.mkdir(embedding_dir)

    parsed_fasta_list = list(SeqIO.parse(protein_fasta_file_with_ID, "fasta"))

    for i in range(len(parsed_fasta_list)):
        this_seq_description = str(parsed_fasta_list[i].description)
        this_sequence = str(parsed_fasta_list[i].seq)

        this_prot_ID = this_seq_description.split("|")[0]
        #sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        # Multimer prediction can be done with chains separated by ':'
        try:

            esm_per_residue_embedding_for_this_prot = generate_esm_per_residue_embedding(this_prot_ID, this_sequence)

            this_prot_embedding_fname = this_prot_ID + ".npy"
            this_prot_embedding_file = os.path.join(embedding_dir, this_prot_embedding_fname)

            np.save(this_prot_embedding_file, esm_per_residue_embedding_for_this_prot)

            #print("{} --> done".format(this_seq_description))
            #print(esm_per_residue_embedding_for_this_prot.shape)

        except:
            
            pass
            #print("something error for {}".format(this_seq_description))