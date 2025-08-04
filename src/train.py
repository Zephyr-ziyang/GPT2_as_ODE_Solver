import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import yaml
import numpy as np
from eval import get_run_metrics
from evaluate_ode import get_paras
from tasks import get_task_sampler
from samplers import get_data_sampler,rand_select_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
import random
from datetime import datetime
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

def eval_step(model, xs, ys, loss_func):
    model.eval()
    with torch.no_grad():
        output = model(xs, ys)
        loss = loss_func(output, ys)
    model.train()
    return loss.detach().item(), output.detach()

def generate_validation_data(data_sampler, task, n_points, bsize, num_tasks, use_h, save_dir):
    """生成并保存验证数据的独立函数"""
    os.makedirs(save_dir, exist_ok=True)
    eval_data = {
        "xs": [],
        "ys": [],
        "metadata": {
            "n_points": n_points,
            "num_tasks": num_tasks,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    for _ in range(num_tasks):
        xs = data_sampler.sample_xs(n_points, bsize, use_h=use_h).cuda()
        ys = task.evaluate(xs, use_h=use_h).cuda()
        eval_data["xs"].append(xs.cpu())
        eval_data["ys"].append(ys.cpu())
    
    # 单文件存储
    torch.save(eval_data, os.path.join(save_dir, f"eval_data_{n_points}.pt"))
    return [d.cuda() for d in eval_data["xs"]], [d.cuda() for d in eval_data["ys"]]


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    firstflag = True
    global task_sampler1, task_sampler2

    curriculum = Curriculum(args.training.curriculum)
    optimizer = torch.optim.Adam(model.parameters(), lr= curriculum.current_lr,weight_decay=args.training.weight_decay)  # add weight decay
    
    
    # 初始化验证数据存储
    current_eval_n_points = None
    current_eval_n_dims = None
    eval_xs_list = []
    eval_ys_list = []

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
            # point_wise_loss_func = task.get_enhanced_metric(steptable)  # loss_func
        else:
            loss_func = task.get_training_metric()  # 损失metric
            # point_wise_loss_func = task.get_metric()  # loss_func
        loss, output = train_step(model, xs, ys, optimizer, loss_func)  # train update 参数。 loss 为总误差

        # 点损失和baseline损失计算
        # point_wise_tags = list(range(curriculum.n_points))
        # point_wise_loss = point_wise_loss_func(output, ys).mean(dim=0)  # 个数据点的逐点平均误差 对bs求mean

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
                    # "pointwise/loss": dict(
                    #     zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    # ),
                    "learning_rate": curriculum.current_lr,
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        if args.training.If_evalloss_in_train and i % args.training.eval_loss_per_steps == 0 and not args.test_run:
            if current_eval_n_points != curriculum.n_points or current_eval_n_dims != curriculum.n_dims_truncated:
                current_eval_n_points = curriculum.n_points
                current_eval_n_dims = curriculum.n_dims_truncated
                
                # 生成验证数据（更新后）
                data_dir = os.path.join(args.out_dir, "valid_data")
                eval_xs_list, eval_ys_list = generate_validation_data(
                    data_sampler, task, 
                    current_eval_n_points, 
                    bsize, 
                    args.training.num_eval_tasks,
                    args.training.If_use_h,
                    data_dir
                )

            # 计算评估损失
            outs = []
            eval_loss = 0.0
            for eval_xs, eval_ys in zip(eval_xs_list, eval_ys_list):
                # 获取steptable
                eval_steptable = task.get_training_steptable(eval_xs,use_h = args.training.If_use_h).cuda()
                # 增强损失 每个数据点计算前几个的损失
                eval_loss_func = task.get_enhanced_training_metric(eval_steptable)
                loss, outputs = eval_step(model, eval_xs, eval_ys, eval_loss_func)  # 使用封装好的eval_step
                eval_loss += loss
                outs.append(outputs)
            
            # 计算平均验证损失并记录
            avg_eval_loss = eval_loss/args.training.num_eval_tasks 
            wandb.log({
                "eval/loss": avg_eval_loss,
                "eval/n_points": current_eval_n_points,  # 添加课程参数记录
                "eval/n_dims": current_eval_n_dims
            }, step=i)

            # 可视化动态，展示模型逐步学习的效果
            if i % 1000 == 0:
                # 可视化样例预测
                if firstflag:
                    print(f"eval_xs_list={eval_xs_list[0].shape}")
                    print(f"eval_ys_list={eval_ys_list[0].shape}")
                    print(f"outs={outs[0].shape}")
                first_xs = eval_xs_list[0]
                first_ys = eval_ys_list[0]
                first_outs = outs[0]
                steps,plot_xs,___ = get_paras(args.training.task, first_xs, first_ys, first_outs, 0, 0)

                plot_xs = plot_xs.cpu().numpy()
                plot_ys = first_ys[0][0].cpu().numpy()
                plot_pred = first_outs[0][0].detach().cpu().numpy()

                if firstflag:
                    print(f"steps={steps}")
                    print(f"plot_xs={plot_xs.shape}")
                    print(f"plot_ys={plot_ys.shape}")
                    print(f"plot_pred={plot_pred.shape}")
                
                plt.figure(figsize=(12, 6))
                plt.plot(plot_xs, plot_ys[:steps], 'b-', label=f'Ground Truth{plot_ys[:steps].shape,plot_xs.shape}')
                plt.plot(plot_xs, plot_pred[:steps], 'r--', label=f'Prediction{plot_pred[:steps].shape}')
                plt.legend()
                save_path = os.path.join(args.out_dir, "validpred_plots", f"pred_plot_{i}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 新增目录创建
                plt.savefig(save_path)
                wandb.log({
                    "predictions": wandb.Image(save_path,
                                            caption=f"Step {i}: Prediction vs Ground Truth")
                }, step=i)
                plt.close()
            model.train()


        # 更新
        curriculum.update()
        optimizer.param_groups[0]['lr'] = curriculum.current_lr
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
        firstflag = False
            

def main(args):
    if args.test_run: # 测试模式 减少规模
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__, # config=args.__dict__ 用于将 args 对象的所有属性（即命令行参数和配置项）以字典形式传递给 WandB 作为配置 
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    # # 在 main 函数中模型初始化后添加：
    # print("=== Model Structure ===")
    # print(model)  # 打印完整模型结构

    # print("\n=== Parameter Names ===")
    # for name, _ in model.named_parameters():
    #     print(name)  # 打印所有参数路径

    # # 临时测试第一个矩阵的维度
    # if hasattr(model, 'h') and len(model.h) > 0:
    #     print("\nFirst layer attention matrix shape:", model.h[0].attn.in_proj_weight.shape)    
    model.cuda()
    model.train()
    # 开始训练
    train(model, args)
    # 训练完成后计算评估指标 并保存至 metrics.json
    # if not args.test_run: #
    #     _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__": 
    parser = QuinineArgumentParser(schema=schema) #
    args = parser.parse_quinfig() # config
    assert args.model.family in ["gpt2", "lstm","gptJ"] #
    print(f"Running with: {args}")

    torch.cuda.set_device(args.set_device) 
    if not args.test_run: # 设置运行id 
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
    # print(args.out_dir)
    
