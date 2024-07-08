import torch
import esm
import argparse
import os
from Bio import SeqIO

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()
#model = model.eval().to(torch.device('cpu'))

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
model.set_chunk_size(64)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-protein_fasta_with_ID", "--protein_fasta_with_ID", type = str, required = True, help = "Protein Fasta File with Protein IDs (required)")
    args = parser.parse_args()

    protein_fasta_file_with_ID = args.protein_fasta_with_ID

    # create PDB directory of predicted structures inside nf work directory
    PDB_dir = 'PDB_dir'
    # Create the directory
    os.mkdir(PDB_dir)

    parsed_fasta_list = list(SeqIO.parse(protein_fasta_file_with_ID, "fasta"))

    for i in range(len(parsed_fasta_list)):
        this_seq_description = str(parsed_fasta_list[i].description)
        this_sequence = str(parsed_fasta_list[i].seq)

        this_prot_ID = this_seq_description.split("|")[0]

        #sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        # Multimer prediction can be done with chains separated by ':'
        try:
            
            with torch.no_grad():
                output = model.infer_pdb(this_sequence)

            this_prot_pdb_fname = this_prot_ID + ".pdb"
            this_output_pdb_fname = os.path.join(PDB_dir, this_prot_pdb_fname)
            
            #print(this_output_pdb_fname)

            with open(this_output_pdb_fname, "w") as f:
                f.write(output)

        except:
            
            #pass
            print("something error for esmfold {}".format(this_seq_description))
