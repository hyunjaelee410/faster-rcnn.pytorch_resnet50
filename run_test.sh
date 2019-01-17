GPU_ID=3
CHECKPOINT="faster_rcnn_1_10_2504.pth"

#CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset pascal_voc --net res50 \
#                   	 --checkpoint $CHECKPOINT --cuda --attention_type style_attention --affine_lr 0.1
#
#CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset pascal_voc --net res50 \
#                   	 --checkpoint $CHECKPOINT --cuda --attention_type style_attention --affine_lr 1.0

CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset pascal_voc --net res50 \
                   	 --checkpoint $CHECKPOINT --cuda --attention_type style_attention --affine_lr 10.0

CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset pascal_voc --net res50 \
                   	 --checkpoint $CHECKPOINT --cuda --attention_type se

CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset pascal_voc --net res50 \
                   	 --checkpoint $CHECKPOINT --cuda
