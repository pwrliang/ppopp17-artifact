export DATASET=/mnt/sda2/liang/delaunay_n24.gr
export THRESHOLD=6073850
./cmake-build-debug/pr -noasync_multi -num_gpus 1 -startwith 1 -single -graphfile $DATASET -semi -sync -cta_np -threshold $THRESHOLD
./cmake-build-debug/pr -noasync_multi -num_gpus 1 -startwith 1 -single -graphfile $DATASET -semi -sync -nocta_np -threshold $THRESHOLD
./cmake-build-debug/pr -noasync_multi -num_gpus 1 -startwith 1 -single -graphfile $DATASET -semi -nosync -cta_np -threshold $THRESHOLD
./cmake-build-debug/pr -noasync_multi -num_gpus 1 -startwith 1 -single -graphfile $DATASET -semi -nosync -nocta_np -threshold $THRESHOLD
./cmake-build-debug/pr -noasync_multi -num_gpus 1 -startwith 1 -single -graphfile $DATASET -cta_np -threshold $THRESHOLD
./cmake-build-debug/pr -noasync_multi -num_gpus 1 -startwith 1 -single -graphfile $DATASET -nocta_np -threshold $THRESHOLD
