import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.nyu.create_dataset import NYUv2
from sam2.build_sam import build_sam2
from sam2.modeling.backbones.tensorlib.utils import (
    mark_only_td_as_trainable,
    print_trainable_parameters,
)
from sam2.sam2_mt import MDMTSAM2Predictor as SAM2
from SIMS import SIMS
from utils import (
    ConfMatrix,
    create_optimizer_scheduler,
    depth_error,
    model_fit,
    normal_error,
)


def parser_args():
    parser = argparse.ArgumentParser(description="SIMS for NYUv2")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--sam_checkpoint",
        default="your checkpoint path",
        type=str,
    )
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l_nyu.yaml")
    parser.add_argument("--data_root", default="your dataset path", type=str)

    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--type", default="None")

    parser.add_argument("--use_quad", default=0, type=int)
    parser.add_argument("--tau", default=-1, type=float)
    parser.add_argument("--lmbd", default=1e-5, type=float)
    parser.add_argument("--K", default=1, type=int)

    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--scheduler", default="linear", type=str)
    parser.add_argument("--warmup_step_ratio", default=0.1, type=float)

    parser.add_argument("--total_epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--accumulate_step", default=1, type=int)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--local-rank", default=-1, type=int)

    return parser.parse_args()


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600),
    )
    return local_rank, torch.device("cuda", local_rank)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloaders(batch_size, data_root):
    train_set = NYUv2(
        root=data_root,
        mode="trainval",
        augmentation=True,
        task="multi-task",
    )
    val_set = NYUv2(
        root=data_root,
        mode="test",
        augmentation=False,
        task="multi-task",
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_predictor(params, device, local_rank):
    sam2 = build_sam2(params.model_cfg, params.sam_checkpoint, device=device)
    domain2task2slice = {
        "NYUD": {
            "seg": slice(0, 13),
            "depth": slice(13, 14),
            "normal": slice(14, None),
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


def collect_predictions(predictor, img, org_dataset):
    seg_pred, dep_pred, normal_pred = None, None, None
    pred = predictor(img, org_dataset, True)
    for domain_task in pred:
        if "seg" in domain_task:
            seg_pred = pred[domain_task].squeeze(2)
        elif "dep" in domain_task:
            dep_pred = pred[domain_task].squeeze(2)
        elif "normal" in domain_task:
            normal_pred = pred[domain_task].squeeze(2)
    return seg_pred, dep_pred, normal_pred


def forward_and_losses(predictor, img, seg_gt, dep_gt, normal_gt, org_dataset):
    seg_pred, dep_pred, normal_pred = collect_predictions(predictor, img, org_dataset)
    seg_pred = F.log_softmax(seg_pred, dim=1)
    normal_pred = normal_pred / torch.norm(normal_pred, p=2, dim=1, keepdim=True)

    loss_seg = model_fit(seg_pred, seg_gt, "semantic")
    loss_dep = model_fit(dep_pred, dep_gt, "depth")
    loss_normal = model_fit(normal_pred, normal_gt, "normal")
    return [loss_seg, loss_dep, loss_normal]


def evaluate(predictor, test_loader, device):
    with torch.no_grad():
        val_dataset = iter(test_loader)
        val_batch = len(test_loader)
        conf_mat_nyud = ConfMatrix(13)
        avg_loss_seg = 0
        avg_loss_dep = 0
        avg_cost_dep = [0, 0]
        avg_cost_normal = [0, 0, 0, 0, 0]

        for _ in range(val_batch):
            val_img, val_seg_gt, val_dep_gt, val_normal_gt = next(val_dataset)
            val_img, val_dep_gt = val_img.to(device), val_dep_gt.to(device)
            val_seg_gt = val_seg_gt.long().to(device)
            val_normal_gt = val_normal_gt.to(device)

            seg_pred, dep_pred, normal_pred = collect_predictions(
                predictor.module,
                val_img,
                "NYUD",
            )
            seg_pred = F.log_softmax(seg_pred, dim=1)
            normal_pred = normal_pred / torch.norm(
                normal_pred,
                p=2,
                dim=1,
                keepdim=True,
            )

            val_loss = [
                model_fit(seg_pred, val_seg_gt, "semantic"),
                model_fit(dep_pred, val_dep_gt, "depth"),
                model_fit(normal_pred, val_normal_gt, "normal"),
            ]
            conf_mat_nyud.update(seg_pred.argmax(1).flatten(), val_seg_gt.flatten())

            avg_loss_seg += val_loss[0].item() / val_batch
            avg_loss_dep += val_loss[1].item() / val_batch
            cost_dep_1, cost_dep_2 = depth_error(dep_pred, val_dep_gt)
            avg_cost_dep[0] += cost_dep_1 / val_batch
            avg_cost_dep[1] += cost_dep_2 / val_batch

            cost_normal = normal_error(normal_pred, val_normal_gt)
            for index, value in enumerate(cost_normal):
                avg_cost_normal[index] += value / val_batch

        avg_cost_seg = [0, 0]
        avg_cost_seg[0], avg_cost_seg[1] = conf_mat_nyud.get_metrics()

    print(
        "NYUD | Test: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f}| "
        "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
            avg_loss_seg,
            avg_cost_seg[0],
            avg_cost_seg[1],
            avg_loss_dep,
            avg_cost_dep[0],
            avg_cost_dep[1],
            avg_cost_normal[0],
            avg_cost_normal[1],
            avg_cost_normal[2],
            avg_cost_normal[3],
            avg_cost_normal[4],
        )
    )


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

    sims = SIMS(
        predictor,
        params,
        device,
        local_rank=local_rank,
        scheduler_builder=create_optimizer_scheduler,
        inner_lr=5e-4,
    )

    for epoch in range(total_epoch):
        start_time = time.time()
        predictor.train()
        train_loader.sampler.set_epoch(epoch)
        train_dataset = iter(train_loader)

        for batch_index in range(train_batch):
            if batch_index % params.accumulate_step == 0:
                optimizer.zero_grad()

            img, seg_gt, dep_gt, normal_gt = next(train_dataset)
            img, dep_gt = img.to(device), dep_gt.to(device)
            seg_gt = seg_gt.long().to(device)
            normal_gt = normal_gt.to(device)
            batch = (img, seg_gt, dep_gt, normal_gt, "NYUD")

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
            
            evaluate(predictor, test_loader, device)

            if epoch % 10 == 0:
                current_state_dict = {
                    name: param
                    for name, param in predictor.named_parameters()
                    if param.requires_grad
                }
                torch.save(
                    current_state_dict,
                    params.output_folder + "/" + str(epoch) + ".pth",
                )

        torch.distributed.barrier()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
