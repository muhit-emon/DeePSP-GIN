# DeePSP-GIN
A Graph Isomorphism Network (GIN)-based deep learning method for identifying and classifying Phage Structural Proteins (PSPs)/Phage Virion Proteins (PVPs)
<br>
<a href="https://doi.org/10.1145/3698587.3701371">Paper Link</a>

# Requirements
<ol>
  <li>Linux operating system</li>
  <li>conda</li>
  <li>GPU (24 GB GPU memory is recommended)</li>
</ol>

# Installation
<pre>
git clone https://github.com/muhit-emon/DeePSP-GIN.git
cd DeePSP-GIN
conda env create -f environment.yml
conda activate DeePSP_GIN
pip install "fair-esm[esmfold]"
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
</pre>

# (1) Binary PSP Classification (PSP Identification)
Go inside <b>DeePSP-GIN</b> directory. <br> <br>
<b>To identify PSPs from an input protein fasta file, use the following command</b> <br>
<pre>
nextflow run binary.nf --prot &ltabsolute/path/to/protein/fasta/file&gt --out_prefix &ltprefix of output csv file name&gt
rm -r work
</pre>
The command line options for this script are: <br><br>
<b>--prot</b>: The absolute path of the fasta file containing protein sequences <br>
<b>--out_prefix</b>: The prefix of the output csv file name <br>

With <b>--out_prefix demo</b>, an output CSV file named <b>demo.csv</b> will be generated inside <b>DeePSP-GIN</b> directory. <br><br>

# (2) Multi-Class PSP Classification
Go inside <b>DeePSP-GIN</b> directory. <br> <br>
<b>To classify PSPs into 7 classess from an input protein fasta file, use the following command</b> <br>
<pre>
nextflow run multi.nf --prot &ltabsolute/path/to/protein/fasta/file&gt --out_prefix &ltprefix of output csv file name&gt
rm -r work
</pre>

# Reference
Muhit Islam Emon, Badhan Das, Ashrith Reddy Thukkaraju, and Liqing Zhang. 2024. DeePSP-GIN: identification and classification of phage structural proteins using predicted protein structure, pretrained protein language model, and graph isomorphism network. In Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (BCB '24). Association for Computing Machinery, New York, NY, USA, Article 42, 1–6. https://doi.org/10.1145/3698587.3701371
