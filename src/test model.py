import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import numpy as np
from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler,rand_select_sampler
from curriculum import Curriculum
from schema import schema
from models import noconf_build_model
import random
import wandb

torch.backends.cudnn.benchmark = True 

# torch.cuda.set_device(2)

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 的 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU，也需要设置


# 设置随机种子
# set_seed(42) # 开局可以固定seeds

def train_step(model, xs, ys, optimizer, loss_func): # 单个批次（batch）的前向传播、损失计算、反向传播和优化器更新
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    global task_sampler1, task_sampler2
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate,weight_decay=args.training.weight_decay)  # add weight decay
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    # 检查checkpoints pt
    if os.path.exists(state_path):
        print("Resuming training from checkpoint")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()
    
    # init sampler
    n_dims = model.n_dims
    bsize = args.training.batch_size
    # 每个step采样一个bs进行训练
    data_sampler = get_data_sampler(args.training.data # gaussion or uniform or odes
    , n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        w_type = args.training.w_type, # 添加w_sample 参数
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )

    pbar = tqdm(range(starting_step, args.training.train_steps))
    # 训练    
    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        # 确保采样的数据直接在 GPU 上
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            use_h = args.training.If_use_h,
            **data_sampler_args,
        ).cuda()
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs, use_h = args.training.If_use_h).cuda()  # ground truth

        # 计算损失
        if args.training.If_enhanced_loss:
            # 获取steptable
            steptable = task.get_training_steptable(xs,use_h = args.training.If_use_h).cuda()
            # 增强损失 每个数据点计算前几个的损失
            loss_func = task.get_enhanced_training_metric(steptable)  # 损失metric
            point_wise_loss_func = task.get_enhanced_metric(steptable)  # loss_func
        else:
            loss_func = task.get_training_metric()  # 损失metric
            point_wise_loss_func = task.get_metric()  # loss_func
        loss, output = train_step(model, xs, ys, optimizer, loss_func)  # train update 参数。 loss 为总误差

        # 点损失和baseline损失计算
        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss = point_wise_loss_func(output, ys).mean(dim=0)  # 个数据点的逐点平均误差 对bs求mean

        # baseline_loss：有效特征维度/数据点数量 表示任务的难度
        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        # 日志记录
        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,  # 相对损失
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )
        # 更新
        curriculum.update()
        # 显示训练进度
        pbar.set_description(f"loss {loss}")
        # 保存模型状态 添加一个文件
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)
        # 保存检查点模型
        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0)  # add weight decay

    modelfamily = "gpt2"
    n_dims = 20
    n_positions = 101
    n_embd = 256
    n_layer = 12
    n_head = 8
    bsize = 64
    n_points = 41

    model = noconf_build_model(modelfamily,n_dims, n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    model.cuda()
    model.train()

    
    


if __name__ == "__main__": 
    main()
    
