

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


lr=1e-4
lambda_=1
scheduler=linear
warmup_step_ratio=0.05
batch_size=1
accumulate_step=1
type=shared_lora
total_epoch=100
sam_checkpoint=your path
data_root=your path

nproc_per_node=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

use_quad=1
tau=3
lmbd=1e-5          


folder_name=city_${lambda_}_${scheduler}_${lr}_${warmup_step_ratio}_${batch_size}_${accumulate_step}_${nproc_per_node}_${total_epoch}_$(date +"%H%M%S_%3N")
mkdir -p outputs/${folder_name}
nohup torchrun --nproc_per_node ${nproc_per_node} --master_port 29611 train_city.py \
        --type ${type} --lr ${lr} --warmup_step_ratio ${warmup_step_ratio} --total_epoch ${total_epoch} --scheduler ${scheduler} --lambda_ ${lambda_} --use_quad ${use_quad} --tau ${tau} --lmbd ${lmbd} \
        --sam_checkpoint ${sam_checkpoint} --data_root ${data_root} \
        --batch_size ${batch_size} --accumulate_step ${accumulate_step} --output_folder outputs/${folder_name}/> \
        outputs/${folder_name}/results.out 2>&1 &
