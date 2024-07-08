# DeePSP-GIN
A Graph Isomorphism Network (GIN)-based deep learning method for identifying and classifying Phage Structural Proteins (PSPs)

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
