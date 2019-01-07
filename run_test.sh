GPU_ID=7
CHECKPOINT="faster_rcnn_1_1_10021.pth"

CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset pascal_voc --net res50 \
                   	 --checkpoint $CHECKPOINT --cuda
