GPU_ID=7
BATCH_SIZE=1
WORKER_NUMBER=1
LEARNING_RATE=0.001
DECAY_STEP=10

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset pascal_voc --net res50 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda
