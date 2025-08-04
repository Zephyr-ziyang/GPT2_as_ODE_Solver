import os
from random import randint
import uuid,json
import matplotlib.pyplot as plt
    
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import numpy as np
from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema_eval import schema
from models import build_model
import random

def evaluate(model, args):
    # 加载模型权重
    model_path = os.path.join(args.out_dir, args.trained_id, "state.pt")
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    
    # 设置模型为评估模式
    model.eval()
    
    # 准备数据
    task_sampler_args = {}

    data = get_data_sampler(args.eval.data, args.eval.n_dims)
    task_sampler = get_task_sampler(args.eval.task, args.eval.n_dims, args.eval.batch_size)
    task = task_sampler(**task_sampler_args)

    xs = data.sample_xs(b_size=args.eval.batch_size, n_points=args.eval.n_points)
    ys = task.evaluate(xs)

    
    # 评估模型
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        xs, ys = xs.cuda(), ys.cuda()
        output = model(xs, ys)
        total += ys.size(0)
    
    # 保存评估结果
    metrics = {
        "loss": loss,
        "accuracy": correct / total,
        "out_put": output[0],
        "xs": xs[0],
        "ys": ys[0]
    }
    
    return metrics

def continuous_evaluate(model, args):
    """
    指生成连续steps的xs
    生成一个batch
    """
    # 加载模型权重
    model_path = os.path.join(args.out_dir, args.trained_id, "state.pt")
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    
    # 设置模型为评估模式
    model.eval()
    
    # 准备数据
    task_sampler_args = {}

    data = get_data_sampler(args.eval.data, args.eval.n_dims)
    task_sampler = get_task_sampler(args.eval.task, args.eval.n_dims, args.eval.batch_size)
    task = task_sampler(**task_sampler_args)

    xs = data.continuous_sample_xs(b_size=args.eval.batch_size, n_points=args.eval.n_points)
    ys = task.evaluate(xs)

    with torch.no_grad():
        xs, ys = xs.cuda(), ys.cuda()
        output = model(xs, ys)
    
    # 保存评估结果
    metrics = {
        "out_put": output,
        "xs": xs,
        "ys": ys,
    }
    
    return metrics

parser = QuinineArgumentParser(schema=schema)
args = parser.parse_quinfig()
assert args.model.family in ["gpt2", "lstm", "gptJ"]

print(f"Running with: {args}")

torch.cuda.set_device(args.set_device) 

model = build_model(args.model)
model.cuda()

if args.eval.evaltype == "common_output":
    metrics = evaluate(model, args)
    # print(f"Evaluation metrics: {metrics}")
elif args.eval.evaltype == "continuous_output":
    metrics = continuous_evaluate(model, args)
    # print(f"Evaluation metrics: {metrics}")

# 保存评估结果路径
eval_path = os.path.join(args.out_dir, args.trained_id,"eval")
if not os.path.exists(eval_path):
    os.makedirs(eval_path)

save_paras_file = os.path.join(eval_path, "paras.json")
if not os.path.exists(save_paras_file):
    with open(save_paras_file, "w") as f:
        json.dump([], f, indent=2)
# print(f"Evaluation metrics: {metrics}")


all_error = []
output = metrics["out_put"]
xs = metrics["xs"]
ys = metrics["ys"]

for i in range(xs.shape[0]):
    # 一个task
    errs = []
    for j in range(xs.shape[1]):
        # 一个step
        eval_y = output[i, j]
        
        if args.eval.task == "ode_ivp_case1":
            steps = int(xs[i, j, 3].item())
            t_eval = torch.linspace(0, 5, steps,device=xs.device)
            #设定参数格式
            paras = {
                "a" : xs[i, 0, 0].item(),
                "b" : xs[i, 0, 1].item(),
                "y_0" : xs[i, 0, 2].item(),
                "t_e" : 5,
                "t_eval": t_eval.cpu().numpy().tolist(),
                "steps" : steps,
                "ground_truth": ys[i, j].cpu().numpy().tolist(),
                "model_eval": eval_y.cpu().numpy().tolist(),
            }

        elif args.eval.task == "ode_ivp_case1plus":
            steps = int(xs[i, j, 4].item())
            t_eval = torch.linspace(0, xs[i, 0, 3], steps,device=xs.device)
            #设定参数格式
            paras = {
                "a" : xs[i, 0, 0].item(),
                "b" : xs[i, 0, 1].item(),
                "y_0" : xs[i, 0, 2].item(),
                "t_e" : xs[i, 0, 3].item(),
                "t_eval": t_eval.cpu().numpy().tolist(),
                "steps" : steps,
                "ground_truth": ys[i, j].cpu().numpy().tolist(),
                "model_eval": eval_y.cpu().numpy().tolist(),
            }

        elif args.eval.task == "ode_ivp_case2" or args.eval.task == "ode_ivp_case2plus":
            steps = int(xs[i, j, 6].item())
            t_eval = torch.linspace(0, xs[i, 0, 5], steps,device=xs.device)
            #设定参数格式
            paras = {
                "a_1" : xs[i, 0, 0].item(),
                "a_2" : xs[i, 0, 1].item(),
                "b_1" : xs[i, 0, 2].item(),
                "b_2" : xs[i, 0, 3].item(),
                "y_0" : xs[i, 0, 4].item(),
                "t_e" : xs[i, 0, 5].item(),
                "t_eval": t_eval.cpu().numpy().tolist(),
                "steps" : steps,
                "ground_truth": ys[i, j].cpu().numpy().tolist(),
                "model_eval": eval_y.cpu().numpy().tolist(),
            }

        
        # print(f"x : ",xs)
        # print(f"y : ",ys)
        # print(f"model_eval : ",eval_y)

        diff = abs(eval_y.cpu().numpy()-ys[i, j].cpu().numpy())
        errs.append(np.mean(diff[:steps]))

        if args.eval.draw_diff and i == 0:
            # 保存图片路径
            filename = f'{args.eval.task}_pointdiff_{j}.png'
            save_path = os.path.join(eval_path,"figs","diff", filename)
            
            # print(save_path)
            if not os.path.exists(os.path.join(eval_path,"figs","diff")):
                os.makedirs(os.path.join(eval_path,"figs","diff"))
                print(f"文件夹创建成功")
            # 清除当前图形
            t_eval_plot = t_eval.cpu().detach().numpy()
            # 保存每张图的参数
            # 从文件中读取现有参数
            with open(save_paras_file, "r") as f:
                try:
                    existing_paras = json.load(f)
                except json.decoder.JSONDecodeError:
                    # 处理文件为空或包含无效JSON的情况
                    existing_paras = []
            existing_paras.append(paras)
            # 将更新后的参数列表写回文件
            with open(save_paras_file, "w") as f:
                json.dump(existing_paras, f, indent=2)

            plt.clf()

            # 绘制图形
            plt.subplot(2, 1, 1)
            plt.plot(t_eval_plot, eval_y.cpu().numpy()[:steps], label='Model')
            plt.plot(t_eval_plot, ys[i, j].cpu().numpy()[:steps], label=f'Ground Truth,steps:{steps}')
            plt.xlabel('t')
            plt.ylabel('y')
            plt.title('model_eval')
            plt.legend()

            # diff
            plt.subplot(2, 1, 2)
            plt.plot(t_eval_plot, diff[:steps], label='Difference')
            plt.xlabel('t')
            plt.ylabel('Error')
            plt.title('Difference between Ground Truth and Model')
            plt.legend()

            plt.tight_layout()
            plt.savefig(save_path)
    all_error.append(errs)

    
# if args.eval.draw_steperr:
#     # 创建新的列表来记录 mean_errs, max_errs, min_errs
#     mean_errs = []
#     max_errs = []
#     min_errs = []

#     print(all_error)

#     # 计算所有 errs 列表在对应位置的均值、最大值和最小值，并添加到相应的列表中
#     for i in range(len(all_error[0])):
#         # 获取所有 errs 列表在当前位置的值
#         values_at_position = [err[i] for err in all_error]
#         # 计算均值、最大值和最小值
#         mean_errs.append(np.mean(values_at_position))
#         max_errs.append(np.max(values_at_position))
#         min_errs.append(np.min(values_at_position))
        
#     # 绘制errs图像
#     steps = np.arange(5, 5+args.eval.n_points)
#     plt.clf()
#     plt.loglog(steps, mean_errs[:args.eval.n_dims-5], label='errors')
#     # plt.plot(steps, max_errs[:args.eval.n_dims-5], label='Max Difference')
#     # plt.plot(steps, min_errs[:args.eval.n_dims-5], label='Min Difference')
#     plt.fill_between(steps, min_errs[:args.eval.n_dims-5], max_errs[:args.eval.n_dims-5], alpha=0.2)
#     plt.xlabel('Steps')
#     plt.ylabel('Error')
#     plt.grid(True)
#     plt.title('Error Change loglog')
#     plt.legend()
#     plt.savefig(os.path.join(eval_path, "err_change.png"))

if args.eval.draw_steperr:
    # 创建新的列表来记录 mean_errs, max_errs, min_errs
    mean_errs = []
    max_errs = []
    min_errs = []

    # print(all_error)

    # 计算所有 errs 列表在对应位置的均值、最大值和最小值，并添加到相应的列表中
    for i in range(len(all_error[0])):
        # 获取所有 errs 列表在当前位置的值
        values_at_position = [err[i] for err in all_error]
        # 计算均值、最大值和最小值
        mean_errs.append(np.mean(values_at_position))
        max_errs.append(np.max(values_at_position))
        min_errs.append(np.min(values_at_position))
        
    # 绘制errs图像
    steps = np.arange(5, 5+args.eval.n_points) # args.eval.n_dims 5+args.eval.n_points
    plt.clf()
    plt.loglog(steps, mean_errs[:args.eval.n_dims-5], label='errors')
    # plt.plot(steps, max_errs[:args.eval.n_dims-5], label='Max Difference')
    # plt.plot(steps, min_errs[:args.eval.n_dims-5], label='Min Difference')
    plt.fill_between(steps, min_errs[:args.eval.n_dims-5], max_errs[:args.eval.n_dims-5], alpha=0.2)
    plt.xlabel('Steps')
    plt.ylabel('Error')
    plt.grid(True)
    plt.title('Error Change loglog')
    plt.legend()
    plt.savefig(os.path.join(eval_path, "err_change.png"))

if args.eval.draw_relativeh_steperr:
    print(f"计算绝对误差（以h为单位的loglog）")
    # 绘制errs图像
    global_errs = []
    # 计算每个步长对应的全局误差
    for i in range(len(all_error[0])):
        values_at_position = [err[i] for err in all_error]
        # 计算 L2 范数作为全局误差
        global_err = np.linalg.norm(values_at_position)
        global_errs.append(global_err)

    # 计算步长
    if args.eval.task == "ode_ivp_case1":
        t_e = 5
    elif args.eval.task == "ode_ivp_case1plus":
        t_e = xs[0, 0, 3].item()
    elif args.eval.task == "ode_ivp_case2" or args.eval.task == "ode_ivp_case2plus":
        t_e = xs[0, 0, 5].item()

    steps = np.arange(5, 5 + args.eval.n_points)
    h = t_e / steps

    # 绘制全局误差随步长的变化
    plt.clf()
    plt.loglog(h, global_errs, label='Global Error (L2 norm)')
    plt.xlabel('Step Size (h)')
    plt.ylabel('Global Error (L2 norm)')
    plt.grid(True)
    plt.title('Global Error Change with Step Size')
    plt.legend()
    plt.savefig(os.path.join(eval_path, "global_err_change.png"))

if args.eval.err_vector_norm:
    print(f"计算相对误差（除以 norm）")
    global_errs = []
    relative_errs = []  # 新增：用于存储相对误差

    # 计算每个步长对应的全局误差
    for i in range(len(all_error[0])):
        values_at_position = [err[i] for err in all_error]
        # 计算 L2 范数作为全局误差
        global_err = np.linalg.norm(values_at_position)
        global_errs.append(global_err)

        # 计算相应解的范数
        true_solution_values = [ys[j, i].cpu().numpy() for j in range(ys.shape[0])]
        true_solution_norm = np.linalg.norm(true_solution_values)

        # 计算相对误差
        relative_err = global_err / true_solution_norm if true_solution_norm != 0 else 0
        relative_errs.append(relative_err)

    # 计算步长
    if args.eval.task == "ode_ivp_case1":
        t_e = 5
    elif args.eval.task == "ode_ivp_case1plus":
        t_e = xs[0, 0, 3].item()
    elif args.eval.task == "ode_ivp_case2" or args.eval.task == "ode_ivp_case2plus":
        t_e = xs[0, 0, 5].item()

    steps = np.arange(5, 5 + args.eval.n_points)
    h = t_e / steps

    # 绘制相对误差随步长的变化
    plt.clf()
    plt.loglog(h, relative_errs, label='Relative Global Error')
    plt.xlabel('Step Size (h)')
    plt.ylabel('Relative Global Error')
    plt.grid(True)
    plt.title('Relative Global Error Change with Step Size')
    plt.legend()
    plt.savefig(os.path.join(eval_path, "relative_global_err_change.png"))

if args.eval.draw_steperr:
    # 创建新的列表来记录 mean_errs, max_errs, min_errs
    mean_errs = []
    max_errs = []
    min_errs = []

    # print(all_error)

    # 计算所有 errs 列表在对应位置的均值、最大值和最小值，并添加到相应的列表中
    for i in range(len(all_error[0])):
        # 获取所有 errs 列表在当前位置的值
        values_at_position = [err[i] for err in all_error]
        # 计算均值、最大值和最小值
        mean_errs.append(np.mean(values_at_position))
        max_errs.append(np.max(values_at_position))
        min_errs.append(np.min(values_at_position))
        
    # 绘制errs图像，跳过最后一个数据
    steps = np.arange(5, 5+args.eval.n_points)[:-1] 
    # 确保 mean_errs, max_errs, min_errs 与 steps 长度一致
    mean_errs = mean_errs[:-1]
    max_errs = max_errs[:-1]
    min_errs = min_errs[:-1]
    plt.clf()
    plt.loglog(steps, mean_errs, label='errors')
    plt.fill_between(steps, min_errs, max_errs, alpha=0.2)
    plt.xlabel('Steps')
    plt.ylabel('Error')
    plt.grid(True)
    plt.title('Error Change loglog (skip_last)')
    plt.legend()

    # 新增拟合代码
    coeffs = np.polyfit(np.log(steps), np.log(mean_errs), 1)
    fit_line = np.exp(coeffs[1]) * steps**coeffs[0]
    plt.loglog(steps, fit_line, 'r--', label=f'Fit (slope={coeffs[0]:.2f})')
    
    plt.xlabel('Steps')
    plt.ylabel('Error')
    plt.grid(True)
    plt.title('Error Change loglog (skip_last)')
    plt.legend()
    
    # 新增斜率标注
    plt.text(0.05, 0.15, 
             f'Slope: {coeffs[0]:.2f}',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

     # 调整纵轴范围以显示更多信息
    plt.ylim(bottom=min(min_errs), top=max(max_errs))

    # 获取正确的起点的横坐标和纵坐标
    start_x = steps[0]
    start_y = mean_errs[0]

    end_x = steps[-1]
    end_y = mean_errs[-1]

    # 获取当前坐标轴的边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 计算起点注释的位置
    start_annotation_x = start_x
    start_annotation_y = start_y * 1.1
    # 检查起点注释的 x 坐标是否越界
    if start_annotation_x < xlim[0]:
        start_annotation_x = xlim[0] * 1.2  # 调整到接近左边界
    elif start_annotation_x > xlim[1]:
        start_annotation_x = xlim[1] * 0.8  # 调整到接近右边界
    # 检查起点注释的 y 坐标是否越界
    if start_annotation_y > ylim[1]:
        start_annotation_y = ylim[1] * 0.9  # 调整到接近顶部

    # 在图上添加起点信息注释
    # 计算起点注释的位置
    start_annotation_x = max(start_annotation_x, ax.get_xlim()[0] * 1.01)  # 确保不超出左边界
    plt.annotate(f'start: ({start_x:.2f}, {start_y:.4f})', 
                xy=(start_x, start_y), 
                xytext=(start_annotation_x, start_annotation_y),
                arrowprops=dict(facecolor='blue', shrink=0.05),
                transform=ax.transData,
                ha='right',  # 添加水平对齐方式
                va='center',
                clip_on=True)

    # 计算终点注释的位置
    end_annotation_x = end_x
    end_annotation_y = end_y * 1.1
    # 检查终点注释的 x 坐标是否越界
    if end_annotation_x < xlim[0]:
        end_annotation_x = xlim[0] * 1.1  # 调整到接近左边界
    elif end_annotation_x > xlim[1]:
        end_annotation_x = xlim[1] * 0.9  # 调整到接近右边界
    # 检查终点注释的 y 坐标是否越界
    if end_annotation_y > ylim[1]:
        end_annotation_y = ylim[1] * 0.9  # 调整到接近顶部

    # 在图上添加终点信息注释
    # 或使用坐标转换（更安全的方式）：
    ax = plt.gca()
    plt.annotate(f'end: ({end_x:.2f}, {end_y:.4f})', 
                xy=(end_x, end_y), 
                xytext=(end_annotation_x, end_annotation_y),  # 调整后的注释文本的位置
                arrowprops=dict(facecolor='red', shrink=0.05),
                transform=ax.transData,
                clip_on=True)


    plt.savefig(os.path.join(eval_path, "err_change_skip_last.png"))


if args.eval.draw_relativeh_steperr:
    print(f"计算绝对误差（以h为单位的loglog）")
    # 绘制errs图像
    global_errs = []
    # 计算每个步长对应的全局误差
    for i in range(len(all_error[0])):
        values_at_position = [err[i] for err in all_error]
        # 计算 L2 范数作为全局误差
        global_err = np.linalg.norm(values_at_position)
        global_errs.append(global_err)

    # 计算步长
    if args.eval.task == "ode_ivp_case1":
        t_e = 5
    elif args.eval.task == "ode_ivp_case1plus":
        t_e = xs[0, 0, 3].item()
    elif args.eval.task == "ode_ivp_case2" or args.eval.task == "ode_ivp_case2plus":
        t_e = xs[0, 0, 5].item()

    steps = np.arange(5, 5 + args.eval.n_points)[:-1]
    h = t_e / steps
    global_errs = global_errs[:-1]
    # 绘制全局误差随步长的变化，跳过最后一个数据
    plt.clf()
    plt.loglog(h, global_errs, label='Global Error (L2 norm)')
    plt.xlabel('Step Size (h)')
    plt.ylabel('Global Error (L2 norm)')
    plt.grid(True)
    plt.title('Global Error Change with Step Size (skip_last)')
    plt.legend()

    # 线性拟合（在log-log空间中）
    coeffs = np.polyfit(np.log(h), np.log(global_errs), 1)
    fit_line = np.exp(coeffs[1]) * h**coeffs[0]

    # 绘制拟合直线
    plt.loglog(h, fit_line, 'r--', label=f'Fit line (slope={coeffs[0]:.2f})')
    plt.legend()  # 更新图例

    # 添加斜率文本标注
    plt.text(0.05, 0.1, 
            f'Slope: {coeffs[0]:.2f}',
            transform=plt.gca().transAxes,  # 使用相对坐标
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))


    # 标注特定点
    # 获取正确的起点的横坐标和纵坐标
    start_x = h[0]
    start_y = global_errs[0]

    # 获取终点的横坐标和纵坐标
    end_x = h[-1]
    end_y = global_errs[-1]
    print(f"终点坐标: ({end_x}, {end_y})")

     # 调整纵轴范围以显示更多信息
    plt.ylim(bottom=min(global_errs), top=max(global_errs))

    # 获取当前坐标轴的边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 计算起点注释的位置
    start_annotation_x = start_x
    start_annotation_y = start_y * 1.1
    # 检查起点注释的 x 坐标是否越界
    if start_annotation_x < xlim[0]:
        start_annotation_x = xlim[0] * 1.2  # 调整到接近左边界
    elif start_annotation_x > xlim[1]:
        start_annotation_x = xlim[1] * 0.8  # 调整到接近右边界
    # 检查起点注释的 y 坐标是否越界
    if start_annotation_y > ylim[1]:
        start_annotation_y = ylim[1] * 0.9  # 调整到接近顶部

    # 在图上添加起点信息注释
    # 计算起点注释的位置
    start_annotation_x = max(start_annotation_x, ax.get_xlim()[0] * 1.01)  # 确保不超出左边界
    plt.annotate(f'start: ({start_x:.2f}, {start_y:.4f})', 
                xy=(start_x, start_y), 
                xytext=(start_annotation_x, start_annotation_y),
                arrowprops=dict(facecolor='blue', shrink=0.05),
                transform=ax.transData,
                ha='right',  # 添加水平对齐方式
                va='center',
                clip_on=True)

    # 计算终点注释的位置
    end_annotation_x = end_x
    end_annotation_y = end_y * 1.1
    # 检查终点注释的 x 坐标是否越界
    if end_annotation_x < xlim[0]:
        end_annotation_x = xlim[0] * 1.1  # 调整到接近左边界
    elif end_annotation_x > xlim[1]:
        end_annotation_x = xlim[1] * 0.9  # 调整到接近右边界
    # 检查终点注释的 y 坐标是否越界
    if end_annotation_y > ylim[1]:
        end_annotation_y = ylim[1] * 0.9  # 调整到接近顶部

    # 在图上添加终点信息注释
    # 或使用坐标转换（更安全的方式）：
    ax = plt.gca()
    plt.annotate(f'end: ({end_x:.2f}, {end_y:.4f})', 
                xy=(end_x, end_y), 
                xytext=(end_annotation_x, end_annotation_y),  # 调整后的注释文本的位置
                arrowprops=dict(facecolor='red', shrink=0.05),
                transform=ax.transData,
                clip_on=True)

    plt.savefig(os.path.join(eval_path, "global_err_change_skip_last.png"))


if args.eval.err_vector_norm:
    print(f"计算相对误差（除以 norm）")
    global_errs = []
    relative_errs = []  # 新增：用于存储相对误差

    # 计算每个步长对应的全局误差
    for i in range(len(all_error[0])):
        values_at_position = [err[i] for err in all_error]
        # 计算 L2 范数作为全局误差
        global_err = np.linalg.norm(values_at_position)
        global_errs.append(global_err)

        # 计算相应解的范数
        true_solution_values = [ys[j, i].cpu().numpy() for j in range(ys.shape[0])]
        true_solution_norm = np.linalg.norm(true_solution_values)

        # 计算相对误差
        relative_err = global_err / true_solution_norm if true_solution_norm != 0 else 0
        relative_errs.append(relative_err)

    # 计算步长
    if args.eval.task == "ode_ivp_case1":
        t_e = 5
    elif args.eval.task == "ode_ivp_case1plus":
        t_e = xs[0, 0, 3].item()
    elif args.eval.task == "ode_ivp_case2" or args.eval.task == "ode_ivp_case2plus":
        t_e = xs[0, 0, 5].item()

    steps = np.arange(5, 5 + args.eval.n_points)[:-1]
    h = t_e / steps

    relative_errs = relative_errs[:-1]

    # 绘制相对误差随步长的变化，跳过最后一个数据
    plt.clf()
    plt.loglog(h, relative_errs, label='Relative Global Error')
    plt.xlabel('Step Size (h)')
    plt.ylabel('Relative Global Error')
    plt.grid(True)
    plt.title('Relative Global Error Change with Step Size (skip_last)')
    plt.legend()

    # 拟合代码
    # 新增拟合代码
    coeffs = np.polyfit(np.log(h), np.log(relative_errs), 1)
    fit_line = np.exp(coeffs[1]) * h**coeffs[0]
    plt.plot(h, fit_line, 'r--', label=f'Fit line (slope={coeffs[0]:.2f})')
    plt.legend()
    
    # 添加斜率标注
    plt.text(0.05, 0.15, 
             f'Slope: {coeffs[0]:.2f}',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))


    # 获取特定点
    # 获取正确的起点的横坐标和纵坐标
    start_x = h[0]
    start_y = relative_errs[0]

    # 获取正确的终点的横坐标和纵坐标
    end_x = h[-1]
    end_y = relative_errs[-1]

    # 调整纵轴范围以显示更多信息
    plt.ylim(bottom=min(relative_errs), top=max(relative_errs))

    # 获取当前坐标轴的边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 计算起点注释的位置
    start_annotation_x = start_x
    start_annotation_y = start_y * 1.1
    # 检查起点注释的 x 坐标是否越界
    if start_annotation_x < xlim[0]:
        start_annotation_x = xlim[0] * 1.2  # 调整到接近左边界
    elif start_annotation_x > xlim[1]:
        start_annotation_x = xlim[1] * 0.8  # 调整到接近右边界
    # 检查起点注释的 y 坐标是否越界
    if start_annotation_y > ylim[1]:
        start_annotation_y = ylim[1] * 0.9  # 调整到接近顶部

    # 在图上添加起点信息注释
    # 计算起点注释的位置
    start_annotation_x = max(start_annotation_x, ax.get_xlim()[0] * 1.01)  # 确保不超出左边界
    plt.annotate(f'start: ({start_x:.2f}, {start_y:.4f})', 
                xy=(start_x, start_y), 
                xytext=(start_annotation_x, start_annotation_y),
                arrowprops=dict(facecolor='blue', shrink=0.05),
                transform=ax.transData,
                ha='right',  # 添加水平对齐方式
                va='center',
                clip_on=True)

    # 计算终点注释的位置
    end_annotation_x = end_x
    end_annotation_y = end_y * 1.1
    # 检查终点注释的 x 坐标是否越界
    if end_annotation_x < xlim[0]:
        end_annotation_x = xlim[0] * 1.1  # 调整到接近左边界
    elif end_annotation_x > xlim[1]:
        end_annotation_x = xlim[1] * 0.9  # 调整到接近右边界
    # 检查终点注释的 y 坐标是否越界
    if end_annotation_y > ylim[1]:
        end_annotation_y = ylim[1] * 0.9  # 调整到接近顶部

    # 在图上添加终点信息注释
    # 或使用坐标转换（更安全的方式）：
    ax = plt.gca()
    plt.annotate(f'end: ({end_x:.2f}, {end_y:.4f})', 
                xy=(end_x, end_y), 
                xytext=(end_annotation_x, end_annotation_y),  # 调整后的注释文本的位置
                arrowprops=dict(facecolor='red', shrink=0.05),
                transform=ax.transData,
                clip_on=True)
    
    plt.savefig(os.path.join(eval_path, "relative_global_err_change_skip_last.png"))