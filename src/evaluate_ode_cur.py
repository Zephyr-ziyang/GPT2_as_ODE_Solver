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
from functools import wraps
from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot

def evaluate_from_model(model, args):
    """
    两种模式，
    1. 生成一个batch的xs,ys
    2. 生成连续steps的xs,ys
    加载模型输出并返回
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

    if args.eval.evaltype == "common_output":
        xs = data.sample_xs(b_size=args.eval.batch_size, n_points=args.eval.n_points, use_h=args.eval.If_use_h)
    elif args.eval.evaltype == "continuous_output":
        xs = data.continuous_sample_xs(b_size=args.eval.batch_size, n_points=args.eval.n_points, use_h=args.eval.If_use_h)
    ys = task.evaluate(xs)

    
    # 评估模型
    with torch.no_grad():
        xs, ys = xs.cuda(), ys.cuda()
        output = model(xs, ys)
    
    # 保存评估结果
    metrics = {
        "out_put": output[0],
        "xs": xs[0],
        "ys": ys[0]
    }
    
    return metrics

def get_paras(taskname, xs, ys, eval_y, i, j):
    """
    输入参数
    taskname: (str) 任务名称
    xs: (torch.tensor([n_points,n_dims])) 输入数据
    ys: (torch.tensor([n_points,n_dims])) 输出数据
    eval_y: (torch.tensor([n_points,n_dims])) 模型输出数据
    i: (int) 第i个task
    j: (int) 第j个prompt
    返回整理好的参数 steps,t_eval,paras
    steps: (int) 步数
    t_eval: (list) 时间点列 = torch.linspace(0, t_e, steps ,device=xs.device)
    paras: (dict) 参数字典
    """

    if not isinstance(xs, torch.Tensor):
        xs = torch.from_numpy(xs).to(eval_y.device)
    if not isinstance(ys, torch.Tensor):
        ys = torch.from_numpy(ys).to(eval_y.device)
    

    if taskname == "ode_ivp_case1":
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

    elif taskname == "ode_ivp_case1plus":
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

    elif taskname == "ode_ivp_case2" or taskname == "ode_ivp_case2plus" or taskname == "ode_ivp_case2vec":
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
    else:
        raise NotImplementedError
    return steps,t_eval,paras

def get_steps(taskname, xs):
    """根据任务类型和输入数据获取steps参数"""
    if taskname == "ode_ivp_case1":
        return xs[:, :, 3].long()  # 返回完整的steps张量
    elif taskname == "ode_ivp_case1plus":
        return xs[:, :, 4].long()
    elif taskname in ["ode_ivp_case2", "ode_ivp_case2plus", "ode_ivp_case2vec"]:
        return xs[:, :, 6].long()
    else:
        raise NotImplementedError

def plot_comparison(t_eval, eval_y, ground_truth, diff, steps, save_path = None, title_suffix=""):
    """绘制模型预测对比图及误差曲线"""
    plt.clf()
    
    # 预测对比子图
    plt.subplot(2, 1, 1)
    plt.plot(t_eval, eval_y, label='Model')
    plt.plot(t_eval, ground_truth, label=f'Ground Truth (steps:{steps})')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(f'Model Evaluation {title_suffix}')
    plt.legend()

    # 误差子图
    plt.subplot(2, 1, 2)
    plt.plot(t_eval, diff, label='Absolute Error')
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.title('Prediction Error')
    plt.legend()
    plt.show()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

def plot_error_analysis(x_values, y_values_list, x_label, y_label, title=None, save_path=None, 
                        labels=None, fit_line=False, annotate_points=False, y_limits=None,
                        skip_last=False, annotate_slope=False, annotate_start_end=False):
    """
    通用误差分析绘图函数
    参数：
    x_values: x轴数据
    y_values_list: 多个y轴数据列表，每个列表对应一个数据集
    x_label: x轴标签
    y_label: y轴标签
    title: 标题
    save_path: 保存路径
    labels: 图例标签列表
    fit_line: 是否拟合曲线
    annotate_points: 是否标注点
    y_limits: y轴范围
    skip_last: 是否跳过最后一个点
    annotate_slope: 是否标注斜率
    annotate_start_end: 是否标注起点和终点
    """
    import matplotlib as mpl
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
    })

    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",  # 添加amsmath宏包
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
    if skip_last:
        x_values = x_values[:-1]
        if isinstance(y_values, dict):
            y_values = {k: v[:-1] for k, v in y_values.items()}
        else:
            y_values = y_values[:-1]
    
    # 存储所有拟合参数
    all_coeffs = []

    plt.clf()
    
    # 颜色循环设置
    colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':']
    
    # 遍历所有数据集
    for idx, y_values in enumerate(y_values_list):
        color = colors[idx % len(colors)]
        line_style = line_styles[(idx // len(colors)) % len(line_styles)]
        
        # 绘制主曲线
        if isinstance(y_values, dict):
            plt.loglog(x_values, y_values['mean'], 
                    color=color, 
                    linestyle=line_style)
            plt.fill_between(...)
        else:
            plt.loglog(x_values, y_values, 
                    color=color,
                    linestyle=line_style)
            y_data = y_values
        
        # 添加拟合线（保持原有逻辑）
        if fit_line:
            coeffs = np.polyfit(np.log(x_values), np.log(y_data), 1)
            all_coeffs.append((coeffs, color))  # 保存颜色信息

            fit_curve = np.exp(coeffs[1]) * x_values**coeffs[0]
            plt.plot(x_values, fit_curve, 
                    color=color,
                    linestyle=':',  # 使用虚线表示拟合线
                    alpha=0.6)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    # 修改x轴显示格式为普通数字
    ax = plt.gca()
    
    # 禁用默认的对数刻度
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
    
    # 设置自定义刻度，每隔5个点显示一个
    step = max(1, len(x_values) // 10)  # 或者根据你的需求调整这个步长
    ax.set_xticks(x_values[::step])
    ax.set_xticklabels([f"{x:.0f}" for x in x_values[::step]])  # 明确设置刻度标签格式
    
    # 禁用次要刻度
    ax.xaxis.set_minor_locator(plt.NullLocator())

    if title is not None:
        plt.title(title)
    plt.legend()
    
    # 设置纵轴范围
    if y_limits:
        plt.ylim(y_limits)
    
    # 添加点标注
    if annotate_points:
        ax = plt.gca()
        for i, (x, y) in enumerate(zip(x_values, y_data)):
            if i % 5 == 0:  # 每隔5个点标注一次
                ax.annotate(f'({x:.1f}, {y:.2e})',
                            xy=(x, y),
                            textcoords='offset points',
                            xytext=(0,10),
                            ha='center')
                
    # 修改斜率标注部分（保留完整信息）
    if annotate_slope and fit_line:
        text_lines = []
        for i, (coeffs, color) in enumerate(all_coeffs):
            label = labels[i] if labels else f'Model {i+1}'
            text_lines.append(
                (f'• {label}: slope={coeffs[0]:.2f}', color)
            )
        
        # 绘制带颜色的多行文本
        y_pos = 0.15
        for text, color in text_lines:
            plt.text(0.05, y_pos, 
                    text,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    color=color,
                    bbox=dict(facecolor='white', alpha=0.8))
            y_pos -= 0.06  # 调整行间距

    # 起点终点标注
    if annotate_start_end and len(x_values) > 1:
        ax = plt.gca()
        points = [
            ('start', 0, 1.1, 'blue'),
            ('end', -1, 1.1, 'red')
        ]

        for label, index, y_scale, color in points:
            x = x_values[index]
            y = y_data[index]
            
            # 计算初始注释位置
            annotation_x = x
            annotation_y = y * y_scale
            
            # 获取当前坐标边界
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # 调整X位置
            if annotation_x < xlim[0]:
                annotation_x = xlim[0] * 1.2
            elif annotation_x > xlim[1]:
                annotation_x = xlim[1] * 0.8
                
            # 调整Y位置
            if annotation_y > ylim[1]:
                annotation_y = ylim[1] * 0.9
            
            # 确保注释在视图范围内
            annotation_x = max(annotation_x, xlim[0] * 1.01)
            annotation_x = min(annotation_x, xlim[1] * 0.99)
            annotation_y = min(annotation_y, ylim[1] * 0.95)
            
            # 添加标注
            ax.annotate(
                f'{label}: ({x:.2f}, {y:.2e})',
                xy=(x, y),
                xytext=(annotation_x, annotation_y),
                arrowprops=dict(
                    facecolor=color,
                    shrink=0.05,
                    width=1,
                    headwidth=5
                ),
                fontsize=10,
                color=color,
                ha='right' if label == 'start' else 'left',
                va='center',
                clip_on=True
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

def calculate_error_stats(all_error):
    """
    统一计算误差统计量
    参数：
    all_error: 包含每个task的误差列表
    返回：
    mean_errs: 每个step的平均误差
    max_errs: 每个step的最大误差
    min_errs: 每个step的最小误差
    """
    mean_errs, max_errs, min_errs = [], [], []
    for i in range(len(all_error[0])):
        values = [err[i] for err in all_error]
        mean_errs.append(np.mean(values))
        max_errs.append(np.max(values))
        min_errs.append(np.min(values))
    return mean_errs, max_errs, min_errs

def generate_equation_params(conf, xs, n_points):
    """根据配置和输入数据生成方程参数和文件名标识"""
    steps_full = np.arange(9, 5 + n_points)
    
    if conf.training.task == "ode_ivp_case1":
        t_e = 5
        equation_title = f"\\frac{{dy}}{{dt}} = {xs[0,0,0].item():.2f}y + {xs[0,0,1].item():.2f},\quad y(0)={xs[0,0,2].item():.2f}"
        filename_params = f"a{xs[0,0,0].item():.1f}_b{xs[0,0,1].item():.1f}_y0{xs[0,0,2].item():.1f}"
    elif conf.training.task == "ode_ivp_case1plus":
        t_e = xs[0, 0, 3].item()
        equation_title = f"\\frac{{dy}}{{dt}} = {xs[0,0,0].item():.2f}y + {xs[0,0,1].item():.2f},\quad y(0)={xs[0,0,2].item():.2f},\quad t_e={t_e:.1f}"
        filename_params = f"a{xs[0,0,0].item():.1f}_b{xs[0,0,1].item():.1f}_y0{xs[0,0,2].item():.1f}_te{t_e:.1f}"
    elif conf.training.data in ["ode_ivp_case2", "ode_ivp_case2plus"]:
        t_e = xs[0, 0, 5].item()
        a1, a2, b1, b2, y0 = (xs[0,0,k].item() for k in [0,1,2,3,4])
        
        equation_title = (
            f"y(t) = e^{{-\\frac{{1}}{{2}}{a1:.2f} t^2 - {a2:.2f} t}} \\left("
            f"{b1:.2f} \\int_0^t e^{{\\frac{{1}}{{2}}{a1:.2f} s^2 + ({a2:.2f}+{b2:.2f})s}} ds + {y0:.2f} \\right)"
        )
        filename_params = (
            f"a1{a1:.1f}_a2{a2:.1f}_"
            f"b1{b1:.1f}_b2{b2:.1f}_"
            f"y0{y0:.1f}_te{t_e:.1f}"
        )
    else:
        raise ValueError("Unsupported task type")

    h_full = t_e / steps_full
    return equation_title, filename_params, h_full, steps_full

def plot_solution(xs_b, ys_pred, ys_true, sample_idx=0, sample_idy=0, title = None, save_path="solution_plot.png"):
    """
    绘制单个样本的预测解与真实解对比
    sample_idx: 要可视化的样本索引
    """
    import matplotlib as mpl
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
    })

    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",  # 添加amsmath宏包
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
    
    # 提取参数
    a1 = xs_b[sample_idx, sample_idy, 0].item()
    a2 = xs_b[sample_idx, sample_idy, 1].item()
    b1 = xs_b[sample_idx, sample_idy, 2].item()
    b2 = xs_b[sample_idx, sample_idy, 3].item()
    t_e = xs_b[sample_idx, sample_idy, 5].item()
    steps = int(xs_b[sample_idx, sample_idy, 6].item())
    
    # 生成时间网格
    t = torch.linspace(0, t_e, steps)
    
    # 提取预测值和真实值
    y_pred = ys_pred[sample_idx, sample_idy, :steps].cpu().numpy()
    y_true = ys_true[sample_idx, sample_idy, :steps].cpu().numpy()

    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(t, y_true, 'b-', linewidth=2, label='Analytic Solution')
    plt.plot(t, y_pred, 'r--', linewidth=1.5, label='Numerical Solution')
    
    # 添加标注
    if title is None:
        title = f"Solution Comparison\n(a1={a1:.2f}, a2={a2:.2f}, b1={b1:.2f}, b2={b2:.2f}, t_e={t_e:.1f}, steps={steps})"
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)
    
    # 保存并关闭
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()

    print(f"Running with: {args}")

    torch.cuda.set_device(args.set_device) 

    model = build_model(args.model)
    model.cuda()

    metrics = evaluate_from_model(model, args)

    # 保存评估结果路径
    eval_path = os.path.join(args.out_dir, args.trained_id,"eval")
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    save_paras_file = os.path.join(eval_path, "paras.json")
    if not os.path.exists(save_paras_file):
        with open(save_paras_file, "w") as f:
            json.dump([], f, indent=2)

    # 转换模型输出结果
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
            steps,t_eval,paras = get_paras(args, xs, ys, eval_y, i, j)

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

                # 绘制图像
                eval_np = eval_y.cpu().numpy()[:steps]
                gt_np = ys[i, j].cpu().numpy()[:steps]
                
                plot_comparison(
                    t_eval=t_eval_plot,
                    eval_y=eval_np,
                    ground_truth=gt_np,
                    diff=diff[:steps],
                    steps=steps,
                    save_path=save_path,
                    title_suffix=args.eval.task  # 可选参数，添加任务名称作为标题后缀
                )
        all_error.append(errs)

    if any([args.eval.draw_steperr, args.eval.draw_relativeh_steperr, args.eval.err_vector_norm]):
        # 公共数据处理部分
        steps_full = np.arange(5, 5 + args.eval.n_points)
        if args.eval.task == "ode_ivp_case1":
            t_e = 5
        elif args.eval.task == "ode_ivp_case1plus":
            t_e = xs[0, 0, 3].item()
        elif args.eval.task in ["ode_ivp_case2", "ode_ivp_case2plus"]:
            t_e = xs[0, 0, 5].item()
        h_full = t_e / steps_full

        # 误差分析主逻辑
        if args.eval.draw_steperr:
            """步长误差分析（标准版）"""
            mean_errs, max_errs, min_errs = calculate_error_stats(all_error)
            plot_error_analysis(
                x_values=steps_full,
                y_values={'mean': mean_errs, 'max': max_errs, 'min': min_errs},
                x_label='Steps',
                y_label='Error',
                title='Error Change loglog',
                save_path=os.path.join(eval_path, "err_change.png"),
                fit_line=True,
                annotate_points=True
            )

            """步长误差分析（跳过末点版）"""
            plot_error_analysis(
                x_values=steps_full,
                y_values={'mean': mean_errs, 'max': max_errs, 'min': min_errs},
                x_label='Steps',
                y_label='Error',
                title='Error Change loglog (skip_last)',
                save_path=os.path.join(eval_path, "err_change_skip_last.png"),
                fit_line=True,
                annotate_slope=True,
                annotate_start_end=True,
                skip_last=True,
                y_limits=(min(min_errs), max(max_errs))
            )

        if args.eval.draw_relativeh_steperr:
            """相对步长误差分析"""
            global_errs = [np.linalg.norm([err[i] for err in all_error]) for i in range(len(all_error[0]))]
            
            # 标准版
            plot_error_analysis(
                x_values=h_full,
                y_values=global_errs,
                x_label='Step Size (h)',
                y_label='Global Error',
                title='Global Error vs Step Size',
                save_path=os.path.join(eval_path, "global_err_change.png"),
                fit_line=True
            )

            # 跳过末点版
            plot_error_analysis(
                x_values=h_full,
                y_values=global_errs,
                x_label='Step Size (h)',
                y_label='Global Error',
                title='Global Error vs Step Size (skip_last)',
                save_path=os.path.join(eval_path, "global_err_change_skip_last.png"),
                fit_line=True,
                annotate_slope=True,
                annotate_start_end=True,
                skip_last=True
            )

        if args.eval.err_vector_norm:
            """向量范数误差分析"""
            global_errs = []
            relative_errs = []
            for i in range(len(all_error[0])):
                vec_norm = np.linalg.norm([err[i] for err in all_error])
                global_errs.append(vec_norm)
                true_norm = np.linalg.norm([ys[j,i].cpu().numpy() for j in range(ys.shape[0])])
                relative_errs.append(vec_norm / true_norm if true_norm != 0 else 0)

            # 标准版
            plot_error_analysis(
                x_values=h_full,
                y_values=relative_errs,
                x_label='Step Size (h)',
                y_label='Relative Error',
                title='Relative Error vs Step Size',
                save_path=os.path.join(eval_path, "relative_global_err_change.png"),
                fit_line=True
            )

            # 跳过末点版
            plot_error_analysis(
                x_values=h_full,
                y_values=relative_errs,
                x_label='Step Size (h)',
                y_label='Relative Error',
                title='Relative Error vs Step Size (skip_last)',
                save_path=os.path.join(eval_path, "relative_global_err_change_skip_last.png"),
                fit_line=True,
                annotate_slope=True,
                annotate_start_end=True,
                skip_last=True
            )

