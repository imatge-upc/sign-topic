#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH --time=24:00:00
#SBATCH --mem 120G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#H2S_ROOT=/mnt/gpid08/users/alvaro.budria/pose2vec/data/OpenPose3D/txt_to_h5/

#H2S_ROOT=../../../data/How2Sign/keypoints
H2S_ROOT=../../../data/How2Sign/text

FAIRSEQ_ROOT=./

python ${FAIRSEQ_ROOT}/prep_how2sign.py --data-root ${H2S_ROOT} --min-n-frames 5 --max-n-frames 5000
