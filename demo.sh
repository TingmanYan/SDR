bin="$(pwd)/build/bin/fdr"
datasetroot="$(pwd)/data"
resultsroot="$(pwd)/results/trainingH"

echo ${bin}

mkdir -p ${resultsroot}
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Adirondack -outputDir ${resultsroot}/Adirondack -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/ArtL -outputDir ${resultsroot}/ArtL -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Jadeplant -outputDir ${resultsroot}/Jadeplant -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Motorcycle -outputDir ${resultsroot}/Motorcycle -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/MotorcycleE -outputDir ${resultsroot}/MotorcycleE -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Piano -outputDir ${resultsroot}/Piano -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/PianoL -outputDir ${resultsroot}/PianoL -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Pipes -outputDir ${resultsroot}/Pipes -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Playroom -outputDir ${resultsroot}/Playroom -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Playtable -outputDir ${resultsroot}/Playtable -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/PlaytableP -outputDir ${resultsroot}/PlaytableP -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Recycle -outputDir ${resultsroot}/Recycle -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Shelves -outputDir ${resultsroot}/Shelves -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Teddy -outputDir ${resultsroot}/Teddy -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
${bin} -targetDir ${datasetroot}/MiddV3/trainingH/Vintage -outputDir ${resultsroot}/Vintage -mode MiddV3 -lambda 0.30 -seg_k 30.0 -inlier_ratio 0.50
