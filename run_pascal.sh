export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


lr=1e-4
lambda_=1
scheduler=linear
warmup_step_ratio=0.05
batch_size=1
accumulate_step=1
type=shared_lora
total_epoch=30
sam_checkpoint=your path
data_root=your path

use_quad=1
tau=3
lmbd=1e-5

nproc_per_node=1
export CUDA_VISIBLE_DEVICES=5


folder_name=2026-0405_pascal_4task_2dim_${lambda_}_${scheduler}_${lr}_${warmup_step_ratio}_${batch_size}_${accumulate_step}_${nproc_per_node}_${total_epoch}_32_shared_lora_DDP_quad_Tlmd$(date +"%H%M%S_%3N")
mkdir -p outputs/${folder_name}
nohup torchrun --nproc_per_node ${nproc_per_node} --master_port 29617 train_pascal.py \
        --type ${type} --lr ${lr} --warmup_step_ratio ${warmup_step_ratio} --total_epoch ${total_epoch} --scheduler ${scheduler} --lambda_ ${lambda_} --use_quad ${use_quad} --tau ${tau} --lmbd ${lmbd} \
        --sam_checkpoint ${sam_checkpoint} --data_root ${data_root} \
        --batch_size ${batch_size} --accumulate_step ${accumulate_step} --output_folder outputs/${folder_name}/> \
        outputs/${folder_name}/results.out 2>&1 &
