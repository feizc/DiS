torchrun --nnodes=1 --nproc_per_node=8 train.py \
--model DiS-S/2 \
--data-path /TrainData/Multimodal/zhengcong.fei/dis/data \
--image-size 32 \
--task-type class-cond \
--num-classes 10