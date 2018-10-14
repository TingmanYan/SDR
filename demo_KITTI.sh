bin="$(pwd)/build/bin/fdr"
datasetroot="$(pwd)/data"
resultsroot="$(pwd)/results/KITTI/disp_0"

echo ${bin}

mkdir -p ${resultsroot}
#${bin} -targetDir ${datasetroot}/KITTI/training -outputDir ${resultsroot} -mode KITTI -lambda 0.30 -seg_k 40.0 -inlier_ratio 0.50 -ndisp 228 -use_swap true
${bin} -targetDir ${datasetroot}/KITTI/training -outputDir ${resultsroot} -mode KITTI -lambda 0.30 -seg_k 80.0 -inlier_ratio 0.50 -ndisp 228 -use_swap false
