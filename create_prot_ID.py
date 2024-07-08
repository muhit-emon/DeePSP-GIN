import argparse
import json
from Bio import SeqIO

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-input_protein_fasta", "--input_protein_fasta", type = str, required = True, help = "Input Protein Fasta File (required)")
    args = parser.parse_args()

    input_protein_fasta_file = args.input_protein_fasta

    f = open("proteins_with_ID.fasta", "a")
    d = {}
    ID = 1
    for sequence in SeqIO.parse(input_protein_fasta_file, "fasta"):
        this_prot_description = str(sequence.description)
        this_prot_seq = str(sequence.seq)

        f.write(">" + str(ID) + "|" + this_prot_description + "\n")
        f.write(this_prot_seq + "\n")
        d[ID] = this_prot_description

        ID+=1

    f.close()

    # Open a file in write mode
    with open('ID_maps_protein_description.json', 'w') as file:
        # Use json.dump() to write the dictionary to the file
        json.dump(d, file)