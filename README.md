# CBaSE
CBaSE is a tool which derives gene-specific probabilistic estimates of the strength of negative and positive selection, for more please check Weghorn &amp; Sunyaev, Nature Genetics [(2017)](https://www.nature.com/articles/ng.3987)

## The file should contain the following four tab-delimited columns in the given order (including a header):

1. Gene symbol
> corresponds to the official gene symbol as used in the UCSC knownGene track
2. Mutation effect
> one of {“missense”, “nonsense”, “coding-synon”}, denoting missense, nonsense (stop-gain and stop-loss), and synonymous mutations, respectively; optionally, 3'-UTR and 5'-UTR mutations (with mutation effects {“utr-3”, “utr-5”}) can be included
3. Alternate allele
> the final allele after the mutation event; one of {A, C, G, T}
4. Context index
> index of the sequence context of the reference allele; 0-based indices of tri- or pentanucleotide contexts can be looked up here


## Running CBaSE
CBaSE can be run in two modes, set by the command line argument model:
- model=0: All six possible models are fitted and model selection is done based on the Akaike information criterion (default).
- model>0: Only the model with index model of the six possible functional forms is fitted.


> By default, all six possible models described in Weghorn & Sunyaev, Nature Genetics (2017) are fitted and model selection is done based on the Akaike information criterion ("Compare all"). Alternatively, q-values under one particular of the six model assumptions can be derived by choosing from the drop-down menu.

> You can choose to compute the sequence context-dependent mutation matrix from either trinucleotide contexts or pentanucleotide contexts. Note that choosing pentanucleotides is only suitable for sufficiently large mutation data sets.

> Use the maf file for head and neck squamous cell carcinoma (HNSC) deposited [here](http://genetics.bwh.harvard.edu/cbase/sample_data_HNSC.txt) as an example for the input format or to try out the algorithm. 

### steps:
- clone or download zipped folder from github

- Command line arguments:

  - pathInputFile: Path to file containing somatic mutation data (see below)
  - auxFolder: Folder containing program and auxiliary files (default "Auxiliary").
  - context: Sequence context size used to compute the cancer-type-specific mutation matrix; one of {0=trinucleotides, 1=pentanucleotides}
  - model: Model assumption for the distribution of expected synonymous mutation counts; one of {0,1,2,3,4,5,6}, where 0 corresponds to the default option that compares all six possible models
  - dataName: Name of the dataset used to identify output files (e.g. cancer type)
### Input file format:


| Gene symbol |	Mutation effect	| Alternate allele	| Context index |
| ----------- |	---------------	| ----------------	| ------------- |
| ECE1 | coding-synon |	C |	26 |
| SAMD11 |missense	| A	| 53 |
| TNFRSF4	| nonsense |	A	| 52 |

- Gene symbol: corresponds to the official gene symbol as used in the UCSC knownGene track.
-  Mutation effect: one of {“missense”, “nonsense”, “coding-synon”}, denoting missense, nonsense (stop-gain and stop-loss), and synonymous mutations, respectively.
-  Alternate allele: the final allele after the mutation event; one of {A, C, G, T}.
-  Context index: index of the sequence context of the reference allele; 0-based indices of tri- or pentanucleotide contexts can be found here.

## Output file format:

CBaSE writes the output, including the q-values for negative and positive selection, into the file "q_values_dataName.txt", located in the folder "Output". The columns contain the following values:

* gene  :  	gene symbol, as provided in the input file
* p_phi_neg:    	p-value of the negative selection signal
* q_phi_neg:    	q-value of the negative selection signal
* phi_neg:    	meta-statistic phi of negative selection
* p_phi_pos:    	p-value of the positive selection signal
* q_phi_pos:    	q-value of the positive selection signal
* phi_pos:    	meta-statistic phi of positive selection
* m_obs:    	observed number of missense mutations
* k_obs:    	observed number of nonsense mutations
* s_obs:    	observed number of synonymous mutations

> To sort by positive (negative) selection signal, sorting by the meta-statistic phi_pos (phi_neg) in decreasing order can break ties when several q-values take on the value zero.


## CBaSE web tool
The CBaSE web tool can be found [here](http://genetics.bwh.harvard.edu/cbase/index.html)

## How to cite
If you find our tool useful, please cite Weghorn & Sunyaev, Nature Genetics [(2017)](https://www.nature.com/articles/ng.3987)
# selection-inference
