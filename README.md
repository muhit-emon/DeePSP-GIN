# DeePSP-GIN
A Graph Isomorphism Network (GIN)-based deep learning method for identifying and classifying Phage Structural Proteins (PSPs)

# Requirements
<ol>
  <li>Linux operating system</li>
  <li>conda</li>
</ol>

# Installation
<pre>
git clone https://github.com/muhit-emon/meta-VF-AMR.git
cd meta-VF-AMR
bash install.sh
conda env create -f environment.yml
</pre>
# conda environment activation
After installation of meta-VF-AMR, a conda environment named <b>meta-VF-AMR</b> will be created.<br>
To activate the environment, run the following command <br>
<pre>
conda activate meta-VF-AMR
</pre>
# (1) Usage on metagenomic paired-end short read data
Go inside <b>meta-VF-AMR/scripts</b> directory. <br> <br>
<b>To run metaVF-AMR on metagenomic paired-end short read data (<span> &#42; </span>.fastq/<span> &#42; </span>.fq/<span> &#42; </span>.fastq.gz/<span> &#42; </span>.fq.gz), use the following command</b> <br>
<pre>
nextflow run pe_pipeline.nf --R1 &ltabsolute/path/to/forward/read/file&gt --R2 &ltabsolute/path/to/reverse/read/file&gt --out_fname &ltprefix of output file name&gt
rm -r work
</pre>
The command line options for this script (<b>pe_pipeline.nf</b>) are: <br><br>
<b>--R1</b>: The absolute path of the fastq file containing forward read sequences <br>
<b>--R2</b>: The absolute path of the fastq file containing reverse read sequences <br>
<b>--out_fname</b>: The prefix of the output file name <br><br>

With <b>--out_fname demo</b>, Four fasta files named <b>demo_Reconstructed_ARG.fasta</b>, <b>demo_Reconstructed_MRG.fasta</b>, <b>demo_Reconstructed_BRG.fasta</b>, and <b>demo_Reconstructed_VF.fasta</b> will be generated inside <b>meta-VF-AMR/TMP</b> directory. <br><br>
