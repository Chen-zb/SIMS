import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.pascal.data.pascal_context import PASCALContext
from datasets.pascal.evaluation.evaluate_utils import PerformanceMeter, get_output
from datasets.pascal.loss_functions import get_loss
from sam2.build_sam import build_sam2
from sam2.modeling.backbones.tensorlib.utils import (
    mark_only_td_as_trainable,
    print_trainable_parameters,
)
from sam2.sam2_mt import MDMTSAM2Predictor as SAM2
from SIMS import SIMS
from utils import create_optimizer_scheduler


TASKS = ["semseg", "human_parts", "sal", "normals"]


def parser_args():
    parser = argparse.ArgumentParser(description="SIMS for Pascal")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--sam_checkpoint",
        default="your checkpoint path",
        type=str,
    )
    parser.add_argument(
        "--model_cfg",
        default="configs/sam2.1/sam2.1_hiera_l_pascal.yaml",
    )
    parser.add_argument("--data_root", default="your dataset path", type=str)

    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--type", default="shared_lora")

    parser.add_argument("--use_quad", default=0, type=int)
    parser.add_argument("--tau", default=3, type=float)
    parser.add_argument("--lmbd", default=1e-5, type=float)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--scheduler", default="linear", type=str)
    parser.add_argument("--warmup_step_ratio", default=0.05, type=float)

    parser.add_argument("--total_epoch", default=30, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--accumulate_step", default=1, type=int)
    parser.add_argument("--output_folder", type=str)

    return parser.parse_args()


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=36000),
    )
    return local_rank, torch.device("cuda", local_rank)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloaders(batch_size, data_root):
    train_database = PASCALContext(
        root=data_root,
        split=["train"],
        aug=True,
        do_edge="edge" in TASKS,
        do_human_parts="human_parts" in TASKS,
        do_semseg="semseg" in TASKS,
        do_normals="normals" in TASKS,
        do_sal="sal" in TASKS,
    )
    test_database = PASCALContext(
        root=data_root,
        split=["val"],
        aug=False,
        do_edge="edge" in TASKS,
        do_human_parts="human_parts" in TASKS,
        do_semseg="semseg" in TASKS,
        do_normals="normals" in TASKS,
        do_sal="sal" in TASKS,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_database)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_database,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_database,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_predictor(params, device, local_rank):
    sam2 = build_sam2(params.model_cfg, params.sam_checkpoint, device=device)
    domain2task2slice = {
        "Pascal": {
            "seg": slice(0, 21),
            "human_parts": slice(21, 28),
            "sal": slice(28, 30),
            "normals": slice(30, None),
        }
    }
    mark_only_td_as_trainable(sam2)

    predictor = SAM2(sam2, domain2task2slice).to(device)
    return DDP(
        predictor,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )


def collect_predictions(predictor, data, org_dataset):
    pred = predictor(data, org_dataset, True)
    outputs = []
    for domain_task in pred:
        if "seg" in domain_task:
            outputs.append(pred[domain_task].squeeze(2))
        elif "human_parts" in domain_task:
            outputs.append(pred[domain_task].squeeze(2))
        elif "sal" in domain_task:
            outputs.append(pred[domain_task].squeeze(2))
        elif "normals" in domain_task:
            outputs.append(pred[domain_task].squeeze(2))
        elif "edge" in domain_task:
            outputs.append(pred[domain_task].squeeze(2))
    return outputs


def forward_and_losses(predictor, train_data, targets, org_dataset, criterion):
    train_pred = collect_predictions(predictor, train_data, org_dataset)
    return [
        criterion[task](train_pred[task_index], targets[task])
        for task_index, task in enumerate(TASKS)
    ]


def evaluate(predictor, test_loader, criterion, avg_cost, epoch):
    with torch.no_grad():
        val_dataset = iter(test_loader)
        val_batch = len(test_loader)
        performance_meter = PerformanceMeter(TASKS)

        for _ in range(val_batch):
            val_batch_data = next(val_dataset)
            val_data = val_batch_data["image"].cuda(non_blocking=True)
            targets = {
                task: val_batch_data[task].cuda(non_blocking=True)
                for task in TASKS
            }
            val_pred = collect_predictions(predictor.module, val_data, "Pascal")

            for task_index, task in enumerate(TASKS):
                avg_cost[epoch, len(TASKS) + task_index] += criterion[task](
                    val_pred[task_index],
                    targets[task],
                ).item()

            performance_meter.update(
                {
                    task: get_output(val_pred[task_index], task)
                    for task_index, task in enumerate(TASKS)
                },
                {task: targets[task] for task in TASKS},
            )

        return performance_meter.get_score(verbose=False)


def main():
    params = parser_args()
    local_rank, device = setup_distributed()
    set_seed(params.seed)

    train_loader, test_loader = build_dataloaders(params.batch_size, params.data_root)
    predictor = build_predictor(params, device, local_rank)

    if dist.get_rank() == 0:
        print_trainable_parameters(predictor)
        print("=== Setting ===")
        for name, value in vars(params).items():
            print(f"{name}: {value}")

    trainable_params = [p for p in predictor.parameters() if p.requires_grad]

    total_epoch = params.total_epoch
    train_batch = len(train_loader)
    params.max_step = train_batch * total_epoch
    params.warmup_step = params.max_step * params.warmup_step_ratio

    optimizer = optim.Adam(trainable_params, lr=params.lr, weight_decay=1e-6)
    scheduler = create_optimizer_scheduler(optimizer, params)

    criterion = {task: get_loss(task).cuda() for task in TASKS}
    avg_cost = torch.zeros([total_epoch, 2 * len(TASKS)])

    sims = SIMS(
        predictor,
        params,
        device,
        local_rank=local_rank,
        scheduler_builder=create_optimizer_scheduler,
    )

    for epoch in range(total_epoch):
        start_time = time.time()
        predictor.train()
        train_loader.sampler.set_epoch(epoch)
        train_dataset = iter(train_loader)

        for batch_index in range(train_batch):
            if batch_index % params.accumulate_step == 0:
                optimizer.zero_grad()

            train_batch_data = next(train_dataset)
            train_data = train_batch_data["image"].cuda(non_blocking=True)
            targets = {
                task: train_batch_data[task].cuda(non_blocking=True)
                for task in TASKS
            }
            batch = (train_data, targets, "Pascal", criterion)

            sims.backward(
                predictor,
                forward_and_losses,
                batch,
                accumulate_step=params.accumulate_step,
            )

            if (batch_index + 1) % params.accumulate_step == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        predictor.eval()
        torch.distributed.barrier()
        if dist.get_rank() == 0:
            print(f"------------------Epoch {epoch}", time.time() - start_time)
            eval_results_test = evaluate(
                predictor,
                test_loader,
                criterion,
                avg_cost,
                epoch,
            )
            print(" Pascal | TEST:", eval_results_test)

        torch.distributed.barrier()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
