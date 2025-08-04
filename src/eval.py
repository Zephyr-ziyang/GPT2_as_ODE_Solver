'''
从配置文件和命令行参数中加载模型及其相关设置。
定义多种数据生成方法和任务评估逻辑。
聚合并保存评估结果，支持不同的任务和策略
'''
import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from samplers import get_data_sampler, sample_transformation,rand_select_sampler


from tasks import get_task_sampler
"""
测试数据中xs_p 策略
类似自回归？模拟逐步暴露的信息量，测试模型对部分已知信息的利用能力和对未知信息的泛化能力。
"""
'''
读取模型运行路径（run_path）下的配置文件（config.yaml）和模型权重。
支持加载最新的权重（state.pt）或特定步骤的模型（model_{step}.pt）。
关键逻辑：
使用 torch.load 加载模型的状态字典，并通过 model.load_state_dict 恢复模型。
'''
def get_model_from_run(run_path, step=-1, only_conf=False, noprint=False):
    config_path = os.path.join(run_path, "config.yaml")
    if not noprint:
        print("run_path:", run_path)

    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp)) # todo 从yaml中读取conf
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


# Functions for evaluation

'''
任务评估 batch
功能：

对一个批次的数据进行模型评估。
如果提供 xs_p，则组合训练数据和测试数据进行逐点评估。
逻辑：
调用 task_sampler 生成任务。
根据模型支持的设备（cuda 或 cpu）调整计算。
如果没有 xs_p，直接评估并计算损失。
如果提供了 xs_p，对测试样本逐点评估，并计算逐点评估指标。
'''
def eval_batch(model, task_sampler, xs, xs_p=None):
    task = task_sampler()
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = "cuda"
    else:
        device = "cpu"

    if xs_p is None:
        ys = task.evaluate(xs)
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task.get_metric()(pred.cpu(), ys)
    else: # 逐点评估，测试数据的每个数据点逐一生成评估样本
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        #
        for i in range(n_points): # xs前i个(已知分布sample)  和xs_p后i个点（通过策略生成）
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
            ys = task.evaluate(xs_comb)

            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach() # inds 指定只对组合样本中的第 i 个数据点进行预测
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

    return metrics


# Functions for generating different kinds of train/test data
# 定义数据生成策略
# 通过不同的生成策略，模拟各种任务难度和特征分布
'''
5种用于生成训练和测试数据的strategy
gen_standard：标准训练数据生成，无特殊变化。
gen_opposite_quadrants：生成符号相反的训练和测试样本。
gen_random_quadrants：生成随机象限的训练样本。
gen_orthogonal_train_test：生成训练集和测试集正交的样本。
gen_overlapping_train_test：生成部分重叠的训练和测试样本。
'''
# 无 xs_p
def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None

# 生成 符号相反 的训练集和测试集
def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post

# 生成 随机象限 的训练集和测试集 每个维度随机符号
def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post

# 生成训练集和测试集，使得测试样本与训练样本的特征在向量空间中是 正交的。
def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i


    return xs_train_pre, xs_test_post



'''
聚合评估结果
功能：
对评估结果metrics进行统计分析，计算均值、标准差和引导法（Bootstrap）置信区间。

输入：
metrics：形状为 [num_eval, n_points] 的张量，表示多个批次的逐点评估结果。
输出： 一个包含以下字段的字典：
mean：逐点的平均值。
std：逐点的标准差。
bootstrap_low 和 bootstrap_high：置信区间的上下界。
'''

def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}

'''
评估整个模型
功能：

执行模型的整体评估，支持多种任务、数据生成策略和配置。
对多个批次的数据调用 eval_batch，并聚合结果。

关键逻辑：
初始化data sampler task_sampler。
动态加载数据生成函数：
generating_func = globals()[f"gen_{prompting_strategy}"]
根据 prompting_strategy 动态选择数据生成策略。
循环执行 eval_batch 并收集结果。
使用 aggregate_metrics 统计和保存结果。
'''
def eval_model( #
    # 参数匹配 kwargs key, 从 kwargs 中解析得到参数
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    If_shift_w_distribution=False, # 默认false ， yaml传入true 启用
    eval_w_type="add",
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    # todo data sampler

    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)

    # todo   (w1+w2)x  if
    if If_shift_w_distribution:
        task_sampler = get_task_sampler(
            task_name, n_dims, batch_size,w_type=eval_w_type, **task_sampler_kwargs
        )
    else:
        task_sampler = get_task_sampler(
            task_name, n_dims, batch_size, **task_sampler_kwargs
        )

    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"] # 根据变量prompting_strategy选择 data生成strategy function
    for i in range(num_eval_examples // batch_size):
        # 根据strategy生成 xs 和 xs_p  (符号相反、随机象限..)
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)

        metrics = eval_batch(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)

'''
自动化评估构建
功能：
根据配置（conf）生成所有支持的评估策略。
包括标准评估、随机象限、正交训练测试等。
用途：
批量管理评估任务，便于扩展和多策略比较。
'''
# todo 根据配置（conf） 读取参数
def build_evals(conf):# 学习 domain shift
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data
    If_shift_w_distribution = conf.training.If_shift_w_distribution
    eval_w_type = conf.eval.eval_w_type
    # 创建评估任务的基础配置，所有任务共享这些参数。
    # 如果具体任务有附加需求，可以在后续阶段覆盖这些参数。
    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
        # todo eval from shifted distribution
        "If_shift_w_distribution":If_shift_w_distribution,
        "eval_w_type": eval_w_type,
    }
    evaluation_kwargs = {}
    # 默认的标准评估任务，其prompting_strategy为"standard"
    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"} #
    #  如果任务名称不是linear_regression：添加一个linear_regression的评估任务，用于与其他任务比较.
    #遍历当前的evaluation_kwargs，将基础参数base_kwargs
    # 合并到每个任务的配置中, 返回更新后的evaluation_kwargs。
    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs # 非linear


    # 生成prompt 的strategy
    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs
"""
return evaluation_kwargs like:
{
    "standard": {
        "task_name": "relu_2nn_regression",
        "n_dims": 20,
        "n_points": 40,
        "batch_size": 64,
        "data_name": "gaussian",
        "prompting_strategy": "standard",
        "eval_w_type": "weight_shift"
    },
    "linear_regression": {
        "task_name": "linear_regression",
        "n_dims": 20,
        "n_points": 40,
        "batch_size": 64,
        "data_name": "gaussian",
        "prompting_strategy": "standard",
        "eval_w_type": "weight_shift"
    }
}
...
"""
def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue

            metrics[model.name] = eval_model(model, **kwargs)
        all_metrics[eval_name] = metrics
    # 保存评估指标
    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False
):
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        model = model.cuda().eval()
        all_models = [model]
        if not skip_baselines: #
            all_models += models.get_relevant_baselines(conf.training.task)
    evaluation_kwargs = build_evals(conf) # 根据conf解析的每个task的参数
    # write result into metrics.json
    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json") # 结果保存路径
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute)
    return all_metrics



def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
            (24, 8): "Transformer-12 layers",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name
'''
运行目录管理
'''

def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            
            params["steps"] = conf.training.train_steps
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    # assert len(df) == len(df.run_name.unique())
    if len(df) != len(df.run_name.unique()):
        print(f"Warning: Found {len(df) - len(df.run_name.unique())} duplicate run_name(s).")

    return df

if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)

            