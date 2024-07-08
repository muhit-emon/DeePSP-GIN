#!/usr/bin/env nextflow
 
// Enable DSL 2 syntax
nextflow.enable.dsl = 2

params.prot = "36.faa" // 36.faa has been used just to initialize. It will be overridden when given as --prot
params.out_prefix = "output" // output has been used just to initialize. It will be overridden when given as --out_prefix

params.path_to_bin_model = "${projectDir}/models/bin.pth"
params.path_to_multi_class_model = "${projectDir}/models/multi_class.pth"

process create_new_prot_ID {

    //publishDir "${projectDir}", mode: "copy"

    input:
    path prot_fa

    output:
    path "proteins_with_ID.fasta", emit: proteins_with_ID
    path "ID_maps_protein_description.json", emit: protein_ID_maps_original_description

    """
    python3 ${projectDir}/create_prot_ID.py --input_protein_fasta ${prot_fa}
    """
}

process esm_per_aa_embedding_generation {

    //publishDir "${projectDir}", mode: "copy"

    input:
    path prot_fa

    output:
    path "esm_embedding_dir", emit: esm_embedding_dir

    """
    python3 ${projectDir}/esm_per_residue_embd_generator.py --protein_fasta_with_ID ${prot_fa}
    """
}

process esmfold {

    //publishDir "${projectDir}", mode: "copy"

    input:
    path prot_fa

    output:
    path "PDB_dir", emit: pdb_dir

    """
    python3 ${projectDir}/esmfold.py --protein_fasta_with_ID ${prot_fa}
    """

}

process create_graph {

    //publishDir "${projectDir}", mode: "copy"

    input:
    path prot_fa
    path esm_embedding_dir
    path pdb_dir

    output:
    path "graph.npy", emit: graph
    path "index_maps_ID.json", emit: index_maps_ID

    """
    python3 ${projectDir}/create_graph_bin.py --protein_fasta_with_ID ${prot_fa} --esm_embedding_dir ${esm_embedding_dir} --pdb_dir ${pdb_dir}
    """
}


process multi_class_prediction {

    publishDir "${projectDir}", mode: "copy"

    input:
    path graph
    path bin_model
    path multi_class_model
    path index_maps_ID
    path id_maps_protein_description
    val output_prefix

    output:
    path "${output_prefix}.csv"

    """
    python3 ${projectDir}/prediction_multi_class.py --graphData ${graph} --bin_model ${bin_model} --multiclass_model ${multi_class_model} --idx_maps_id ${index_maps_ID} --id_maps_prot_des ${id_maps_protein_description} --output_prefix ${output_prefix}
    """

}

workflow {

	input_protein_ch = Channel.from(params.prot) // this channel contains input protein fasta file
    
    protein_fasta_with_ID_ch = create_new_prot_ID(input_protein_ch)

    esm_ch = esm_per_aa_embedding_generation(protein_fasta_with_ID_ch.proteins_with_ID)
    esmfold_ch = esmfold(protein_fasta_with_ID_ch.proteins_with_ID)

    graph_ch = create_graph(protein_fasta_with_ID_ch.proteins_with_ID, esm_ch.esm_embedding_dir, esmfold_ch.pdb_dir)

    multi_class_prediction(graph_ch.graph, params.path_to_bin_model, params.path_to_multi_class_model, graph_ch.index_maps_ID, protein_fasta_with_ID_ch.protein_ID_maps_original_description, params.out_prefix)
	
}