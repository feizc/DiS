torchrun --nnodes=1 --nproc_per_node=8 train.py \
--model DiS-L/2 \
--dataset-type imagenet \
--data-path /TrainData/Multimodal/public/datasets/ImageNet \
--image-size 256 \
--task-type class-cond \
--num-classes 1000 