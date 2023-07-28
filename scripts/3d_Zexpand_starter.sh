#!/bin/bash

gpds=(
71
72
73
74
75
)

stencils=(
"D3Q15"
"D3Q19"
"D3Q27"
)

dpzs=(
"0.05" # 1 node
"0.15" # 3 nodes
"1"    # equals gpd nodes
"19"   # equals dpy*gpd nodes
)


for stencil in ${stencils[@]}; do
    for dpz in ${dpzs[@]}; do
        sbatch --job-name "3d_Zexpand_v01_${stencil}_dpz${dpz}" 3d_Zexpand_jobScript.sh ${stencil} ${dpz} 
    done
done
