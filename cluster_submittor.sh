#!/bin/sh
#$ -N ChimpSimilar
#$ -cwd
#$ -q short-sl7,guest
#$ -l virtual_free=4G
#$ -pe ompi 1
#$ -o error_files/ChimpSimilar/
#$ -e error_files/ChimpSimilar/


#cd /nfs/users/dweghorn/hali/sandbox/CBaSE_matrix_each_gene/;
module load Python/3.9.5-GCCcore-10.3.0;
#loop from 0 to 9
for i in {0..9}; do
    #job id is $SGE_TASK_ID + $i
    jobid=$(($SGE_TASK_ID + $i));
    echo "Job ID: $jobid";
    python3.9 main.py $jobid;
done

echo "Done";
#qsub -t 1-30880:10 cluster_submittor.sh