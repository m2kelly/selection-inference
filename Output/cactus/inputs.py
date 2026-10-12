maf_file ='/home/maria/effective_pop/data/wg_primate.maf'
external_matrix ='/home/maria/find_intron_matrix/output_cactus_nocpg_nodup_anc4/rates'
output_name = 'cactus'
output_directory = 'Output_cactus_HCLCA_qvalues'

coding = False
#analysis_pairs = [{'target':'hg38', 'reference':'Anc4'},{'target':'Anc4', 'reference':'Anc3'}]
analysis_pairs = [{'target':'hg38', 'reference':'Anc4'}]



coord_spcs = 'hg38'
annotations_file = '/home/maria/hossam_cbase_exons/exon_merged_ids.gtf'
#must be 1 based gtf file 

region_coding_dict = {'first_coding_exon':True, 'internal_exon':True,
                      'last_coding_exon':True, 'one_exon':True,
                      'first_of_twos':True, 'second_of_twos':True,}



groups_to_collapse = [ ['first_coding_exon', 'first_of_twos',
                       'last_coding_exon', 'second_of_twos',
                       'internal_exon', 'one_exon']]


collapse_threshold = 999999999 #minimim number of mutations in the matrix, otherwise collapse

gene_min_size = 100

excluded_regions = {'/home/maria/hossam_cbase_exons/cpg_dup.gtf': ['merged_prohibited']}




'''analysis_pairs = [{'reference':'HCLCA', 'target':'homo_sapiens'},
                  {'reference':'HCGLCA', 'target':'HCLCA'},
                  {'reference':'HCGOrLCA', 'target':'HCGLCA'},
                  {'reference':'HCGOrGib', 'target':'HCGOrLCA'},
                  {'reference':'HCGLCA', 'target':'homo_sapiens'},
                  {'reference':'HCGOrGib', 'target':'HCGLCA'},
                  ]'''

