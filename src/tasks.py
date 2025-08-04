import math
from scipy.integrate import simpson,trapz,romberg,quad
from scipy.integrate import solve_ivp
import torch
import numpy as np

# 优化效率Para部分使用的并行化计算
import multiprocessing
from functools import partial

# 常用的评估函数
def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)

# 抽象基类，所有具体任务（如线性回归、分类等）都从它继承
class Task:
    def __init__(self, n_dims, batch_size,w_type="gaussian", pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict # 任务参数池
        self.seeds = seeds
        self.w_type = w_type
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks,w_type):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_enhanced_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError
    
    @staticmethod
    def get_enhanced_training_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_steptable():
        raise NotImplementedError

# w sampler
def get_task_sampler(
    task_name, n_dims, batch_size,w_type="gaussian", pool_dict=None, num_tasks=None, **kwargs
):
    ode_task_names_to_classes = {
        "ode_ivp_case1": ODEIVPCase1,
        "ode_ivp_case1plus": ODEIVPCase1plus,
        "ode_ivp_case2": ODEIVPCase2,
        "ode_ivp_case2vec": ODEIVPCase2vec,
        "ode_ivp_case2plus": ODEIVPCase2plus,
        "ode_ivp_case2BSL": ODEIVPCase2BSL,
        "ode_ivp_case1plusBSL": ODEIVPCase1plusBSL,
        # "ode_ivp_case2Para": ODEIVPCase2Para,
    }
    if task_name in ode_task_names_to_classes:
        task_cls = ode_task_names_to_classes[task_name]
        return lambda **args: task_cls(n_dims, batch_size,pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError

class ODEIVPCase1(Task):
    """
    The problem is 
    dy/dt = ay+b,
    y(t0) = t0,
    t\in [t0,t1]
    要求x格式  x[i,j] = [a,b,y0,steps,0,0,0,...,0]
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """因为 dy/dt=ay+b中所有的数据都在x中传递了,没有其他的参数,所以这里的n_dims就是a和b的维度"""
        super(ODEIVPCase1,self).__init__(n_dims, batch_size, pool_dict, seeds)
        # (b_size, n_points, n_dims)
        self.t_0, self.t_e = 0, 5
 
    def evaluate(self, xs_b):
        ground_truth = lambda a, b, y0, t: (y0 + b / a) * np.exp(a * t) - b / a
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], self.n_dims, device=xs_b.device)
        for i in range(xs_b.shape[0]):
            for j in range(xs_b.shape[1]):
                a = xs_b[i, j, 0]
                b = xs_b[i, j, 1]
                y0 = xs_b[i, j, 2]
                steps = int(xs_b[i, j, 3].item())
                t = np.linspace(self.t_0, self.t_e, steps)
                ys_b[i, j] = torch.cat([ground_truth(a, b, y0, t),torch.zeros(self.n_dims - steps)],dim=0)
        return ys_b
    
    @staticmethod
    def get_metric(self):
        return squared_error
    @staticmethod
    def get_training_metric(self):
        
        return mean_squared_error

class ODEIVPCase2plus(Task):
    """
    与原版的区别在于使用了梯形公式计算积分,这样精度变为了10-3
    The problem is 
    dy/dt + p(t)y = q(t),
    p(t) = a_1*t+a_2,
    q(t) = b_1*e^(b_2*t),
    y(t0) = y_0,
    t\in [t_0,t_e],
    要求x格式 x[i,j]=[a_1,a_2,b_1,b_2,y_0,t_e,steps,0,...,0]
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """因为 dy/dt=ay+b中所有的数据都在x中传递了,没有其他的参数,所以这里的n_dims就是a和b的维度"""
        super(ODEIVPCase2plus,self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.t_0 = 0
 
    def evaluate(self, xs_b):
        # def ground_truth_tensor(argss):
        #     """
        #     计算变上限积分函数 y(t) 在 [t_0, t_e] 区间内 steps 个均匀分点上的值。
        #     返回形状为 (steps) 的张量
        #     """
        #     # 解包参数，保持批量维度
        #     a_1, a_2, b_1, b_2, y_0, t_e, steps = argss[:, 0], argss[:, 1], argss[:, 2], argss[:, 3], argss[:, 4], argss[:, 5], argss[:, 6]
        #     steps = steps.long()

        #     # 创建被积函数（保持张量操作）
        #     def integrand(s, a_1, a_2, b_1, b_2):
        #         return b_1 * torch.exp((a_1 / 2) * s**2 + (a_2 + b_2) * s)

        #     max_steps = steps.max().item()
        #     y_values = torch.zeros(argss.shape[0], max_steps, device=argss.device)

        #     for idx in range(argss.shape[0]):
        #         new_t_values = torch.linspace(0, t_e[idx], steps[idx], device=argss.device)
        #         inbracket = y_0[idx]
        #         y_values[idx, 0] = inbracket

        #         for k in range(1, steps[idx]):
        #             t = new_t_values[k]
        #             t_slice = torch.linspace(new_t_values[k - 1], t, 3, device=argss.device)
        #             integrand_slice = integrand(t_slice, a_1[idx], a_2[idx], b_1[idx], b_2[idx])

        #             h = (t - new_t_values[k - 1]) / 2
        #             integral_result = (h / 3) * (integrand_slice[0] + 4 * integrand_slice[1] + integrand_slice[2])

        #             inbracket += integral_result
        #             y_values[idx, k] = torch.exp(-((a_1[idx] / 2) * t**2 + a_2[idx] * t)) * inbracket

        #     return y_values

        # self.steps = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)

        # batch_size = xs_b.shape[0]
        # num_points = xs_b.shape[1]
        # argss = xs_b[:, :, :7].reshape(-1, 7)
        # ys = ground_truth_tensor(argss)
        # ys = ys.reshape(batch_size, num_points, -1)

        # ys_b = torch.zeros(batch_size, num_points, self.n_dims, device=xs_b.device)
        # steps = xs_b[:, :, 6].long()
        # for i in range(batch_size):
        #     for j in range(num_points):
        #         num_steps = steps[i, j]
        #         ys_b[i, j, :num_steps] = ys[i, j, :num_steps]

        # return ys_b
        def ground_truth_tensor(argss):
            """
            计算变上限积分函数 y(t) 在 [t_0, t_e] 区间内 steps 个均匀分点上的值。
            """
            # 解包参数，保持批量维度
            a_1, a_2, b_1, b_2, y_0, t_e, steps = argss[0], argss[1], argss[2], argss[3], argss[4], argss[5], argss[6]
            steps = steps.long()

            # 创建被积函数
            def integrand(s):
                return b_1 * torch.exp((a_1 / 2) * s**2 + (a_2 + b_2) * s,device=argss.device)
            
            # 创建新的分点用于计算 y(t)
            new_t_values = torch.linspace(0, t_e, steps)
            y_values = []

            # 先计算括号中的函数值，使用simps简化计算
            inbracket = y_0
            y_values.append(inbracket)
            pre_t = 0
            for t in new_t_values[1:]:
                # 总共有steps-1个需要积分计算的区间[pre_t,t]
                # 获取 [pre_t,t] 的分点和对应的被积函数值(3个分点 - 11个分点有点恐怖了...300个小时)

                t_slice = torch.linspace(pre_t, t, 3)
                integrand_slice = integrand(t_slice)

                # 使用辛普森法则计算五点积分
                integral_result = trapz(integrand_slice, x=t_slice)
                
                inbracket += integral_result

                # 计算 y(t)
                y_t = torch.exp(-((a_1 / 2) * t**2 + a_2 * t)) * inbracket
                y_values.append(y_t)
                pre_t = t

            
            return torch.tensor(y_values)

        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], self.n_dims, device=xs_b.device)
        for i in range(xs_b.shape[0]):
            for j in range(xs_b.shape[1]):
                steps = int(xs_b[i, j, 6].item())
                ys_b[i, j] = torch.cat([ground_truth_tensor(xs_b[i, j, :7]),torch.zeros(self.n_dims - steps)],dim=0)
        return ys_b
    
    @staticmethod
    def get_metric(self):
        # self.steps 
        def error_sq(ys, ys_pred):
            # ys: (b_size, n_points, n_dims)
            # ys_pred: (b_size, n_points, n_dims)
            # return: (b_size, n_points, n_dims)

            return (ys[:,:,:self.steps] - ys_pred[:,:,:self.steps]).square()
        return error_sq
    @staticmethod
    def get_training_metric(self):
        def error_msq(ys, ys_pred):
            # ys: (b_size, n_points, n_dims)
            # ys_pred: (b_size, n_points, n_dims)
            # return: (b_size, n_points, n_dims)
            return (ys[:,:,:self.steps] - ys_pred[:,:,:self.steps]).square().mean()
        return error_msq

class ODEIVPCase1plus(Task):
    """
    The problem is 
    dy/dt = ay+b,
    y(t0) = t0,
    t\in [t_0,t_ed],
    要求x格式  x:[a,b,y0,t_ed,steps,0,0,0,...,0]
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """因为 dy/dt=ay+b中所有的数据都在x中传递了,没有其他的参数,所以这里的n_dims就是a和b的维度"""
        super(ODEIVPCase1plus,self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.t_0 = 0

    def evaluate(self, xs_b, use_h = False):
        # 使用 PyTorch 函数替代 NumPy 的指数函数
        ground_truth = lambda a, b, y0, t: (y0 + b / a) * torch.exp(a * t) - b / a
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], self.n_dims, device=xs_b.device)
        for i in range(xs_b.shape[0]):
            a = xs_b[i, 0, 0]
            b = xs_b[i, 0, 1]
            y0 = xs_b[i, 0, 2]
            t_ed = xs_b[i, 0, 3]
            for j in range(xs_b.shape[1]):
                # 使用 torch.linspace 替代 np.linspace
                if use_h:
                    h = xs_b[i, j, 4].item()  # 获取步长h
                    steps = int((t_ed - self.t_0) / h) + 1  # 计算总步数
                    assert steps <= self.n_dims,f"Error: t = {t}, steps = {steps}, h = {h}"
                    t = torch.linspace(self.t_0, self.t_0+(h*(steps-1)), steps, device=xs_b.device)
                else:
                    steps = int(xs_b[i, j, 4].item())
                    t = torch.linspace(self.t_0, t_ed, steps, device=xs_b.device)
                
                y_values = ground_truth(a, b, y0, t)
                zeros = torch.zeros(self.n_dims - len(t), device=xs_b.device)
                ys_b[i, j] = torch.cat([y_values, zeros], dim=0)
        return ys_b
    
    @staticmethod
    def get_metric():
        return squared_error
    @staticmethod
    def get_training_metric():
        return mean_squared_error
    @staticmethod
    def get_enhanced_metric(step_table):
        # self.steps 
        def error_sq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            return torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
        return error_sq
    @staticmethod
    def get_enhanced_training_metric(step_table):
        def error_msq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            squared_errors = torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
            return squared_errors.mean()
        return error_msq
    
    @staticmethod
    def get_training_steptable(xs_b,use_h = False):
        # 直接提取步数信息
        if use_h:
            return ((xs_b[..., 3] / xs_b[..., 4]) + 1).long()
        return xs_b[..., 4].long()

class ODEIVPCase1plusBSL(Task):
    """
    The problem is 
    dy/dt = ay+b,
    y(t0) = t0,
    t\in [t_0,t_ed],
    要求x格式  x:[a,b,y0,t_ed,steps,0,0,0,...,0]
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """因为 dy/dt=ay+b中所有的数据都在x中传递了,没有其他的参数,所以这里的n_dims就是a和b的维度"""
        super(ODEIVPCase1plusBSL,self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.t_0 = 0

    def evaluate(self, xs_b, method = 'RK45'):
        """
        methods = [
            'euler_explicit',  # 显式欧拉方法
            'euler_implicit',  # 隐式欧拉方法
            'euler_improved',  # 改进欧拉方法
            'euler_adaptive',  # 自适应欧拉方法
            'RK45',     # (默认) 显式Runge-Kutta方法 (4,5阶)，适用于非刚性问题
            'RK23',     # 显式Runge-Kutta方法 (2,3阶)，适用于精度要求不高的问题
            'DOP853',   # 高阶显式Runge-Kutta方法 (8阶，13阶精度)，适合高精度需求
            'Radau',    # 隐式Runge-Kutta方法，适用于刚性问题
            'BDF',      # 隐式后向差分公式，适用于刚性问题
            'LSODA'     # 自动切换刚性/非刚性求解器 (基于LSODE算法)
        ]
        """
        params = xs_b[..., :5].cpu().numpy()
        batch_size, num_points, _ = params.shape
        
        all_y_values = []
        
        for a, b, y0, t_e, steps in params.reshape(-1, 5):
            steps = int(steps)
            t = np.linspace(0, t_e, steps)
            h = t[1] - t[0] if steps > 1 else t_e
            
            # 新增欧拉方法选择逻辑
            if method.startswith('euler'):
                # 显式欧拉
                if method == 'euler_explicit':
                    y_values = [y0]
                    for _ in range(steps-1):
                        y_next = y_values[-1] + h * (a * y_values[-1] + b)
                        y_values.append(y_next)
                
                # 隐式欧拉
                elif method == 'euler_implicit':
                    y_values = [y0]
                    for _ in range(steps-1):
                        # 解析求解隐式方程 y_{n+1} = y_n + h*(a y_{n+1} + b)
                        y_next = (y_values[-1] + h*b) / (1 - a*h)
                        y_values.append(y_next)
                
                # 改进欧拉（预测-校正）
                elif method == 'euler_improved':
                    y_values = [y0]
                    for _ in range(steps-1):
                        y_p = y_values[-1] + h * (a * y_values[-1] + b)
                        y_next = y_values[-1] + h/2 * (
                            (a * y_values[-1] + b) + (a * y_p + b)
                        )
                        y_values.append(y_next)
                
                # 自适应欧拉（简化的变步长）
                elif method == 'euler_adaptive':
                    y_values = [y0]
                    t_current = 0
                    while t_current < t_e - 1e-6:
                        # 动态调整步长
                        h = min(h, t_e - t_current)
                        # 双步长计算
                        y_step1 = y_values[-1] + h * (a * y_values[-1] + b)
                        y_step2 = y_values[-1] + h/2 * (a * y_values[-1] + b)
                        y_step2 = y_step2 + h/2 * (a * y_step2 + b)
                        
                        # 误差估计
                        error = np.abs(y_step1 - y_step2)
                        h = 0.9 * h * (1e-4 / error)**0.5  # 简化的自适应逻辑
                        
                        y_values.append(y_step2)
                        t_current += h
                    
                    # 插值到指定步数
                    y_values = np.interp(t, np.linspace(0, t_e, len(y_values)), y_values)
                
                else:
                    raise ValueError(f"Unsupported Euler method: {method}")
            
            # 原有方法保持不变
            else:
                def ode_func(t, y):
                    return a * y + b
                
                sol = solve_ivp(ode_func, [0, t_e], [y0],
                              method=method, t_eval=t)
                y_values = sol.y[0]

            all_y_values.append(torch.tensor(y_values, device=xs_b.device))

        # 填充结果张量
        ys_b = torch.zeros(batch_size, num_points, self.n_dims, device=xs_b.device)
        steps_tensor = xs_b[..., 4].long()
        
        for idx, (i, j) in enumerate(np.ndindex(batch_size, num_points)):
            valid_steps = min(steps_tensor[i,j], self.n_dims)
            ys_b[i, j, :valid_steps] = all_y_values[idx][:valid_steps]
            
        return ys_b
    
    @staticmethod
    def get_metric():
        return squared_error
    @staticmethod
    def get_training_metric():
        return mean_squared_error
    @staticmethod
    def get_enhanced_metric(step_table):
        # self.steps 
        def error_sq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            return torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
        return error_sq
    @staticmethod
    def get_enhanced_training_metric(step_table):
        def error_msq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            squared_errors = torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
            return squared_errors.mean()
        return error_msq
    
    @staticmethod
    def get_training_steptable(xs_b,use_h = False):
        # 直接提取步数信息
        if use_h:
            return ((xs_b[..., 3] / xs_b[..., 4]) + 1).long()
        return xs_b[..., 4].long()

class ODEIVPCase2(Task):
    """
    The problem is 
    dy/dt + p(t)y = q(t),
    p(t) = a_1*t+a_2,
    q(t) = b_1*e^(b_2*t),
    y(t0) = y_0,
    t\in [t_0,t_e],
    要求x格式 x[i,j]=[a_1,a_2,b_1,b_2,y_0,t_e,steps,0,...,0]
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """因为 dy/dt=ay+b中所有的数据都在x中传递了,没有其他的参数,所以这里的n_dims就是a和b的维度"""
        super(ODEIVPCase2,self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.t_0 = 0
        self.steps = torch.zeros(1) # 在evaluate中将改为一个大小为(b_size, n_points)的张量，内容对应于每个xs点的步数，每次evaluate(即计算ys)的时候将会更新
    def evaluate(self, xs_b, use_h = False):
        # 批量转换所有参数到CPU numpy数组
        params = xs_b[..., :7].cpu().numpy()  # shape [batch, points, 7]
        batch_size, num_points, _ = params.shape
        np.save("params_debug.npy", params)
        torch.save(xs_b[..., :7], "params_tensor.pt")
        # 预分配结果数组
        all_y_values = []
        for a1, a2, b1, b2, y0, te, steps in params.reshape(-1, 7):
            steps = int(steps)
            t = np.linspace(0, te, steps)
            
            # ==== 修改的积分计算部分 ====
            current_integral = 0.0
            y_values = [y0]
            pre_t = 0.0
            
            for curr_t in t[1:]:  # 从第二个时间点开始迭代
                # 计算当前区间的积分
                integral, _ = quad(
                    lambda s: b1 * np.exp((a1/2)*s**2 + (a2 + b2)*s),
                    pre_t, curr_t,
                    epsabs=1e-12,
                    epsrel=1e-12,
                    limit = 100
                )
                current_integral += integral
                
                # 计算当前时间点的y值
                y_t = np.exp(-(a1/2)*curr_t**2 - a2*curr_t) * (y0 + current_integral)
                y_values.append(y_t)
                pre_t = curr_t
            # ==== 修改结束 ====
            
            full_y = np.array(y_values)
            all_y_values.append(torch.tensor(full_y, device=xs_b.device))
        
        # 重塑结果形状并填充
        ys_b = torch.zeros(batch_size, num_points, self.n_dims, device=xs_b.device)
        steps = xs_b[..., 6].long()
        
        for idx, (i, j) in enumerate(np.ndindex(batch_size, num_points)):
            ys_b[i, j, :steps[i,j]] = all_y_values[idx]
            
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    @staticmethod
    def get_enhanced_metric(step_table):
        # self.steps 
        def error_sq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            return torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
        return error_sq

    @staticmethod
    def get_enhanced_training_metric(step_table):
        def error_msq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            squared_errors = torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
            return squared_errors.mean()
        return error_msq

    @staticmethod
    def get_training_steptable(xs_b):
        # 直接提取步数信息
        return xs_b[..., 6].long()

class ODEIVPCase2BSL(Task):
    """
    The problem is 
    dy/dt + p(t)y = q(t),
    p(t) = a_1*t+a_2,
    q(t) = b_1*e^(b_2*t),
    y(t0) = y_0,
    t\in [t_0,t_e],
    要求x格式 x[i,j]=[a_1,a_2,b_1,b_2,y_0,t_e,steps,0,...,0]
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """因为 dy/dt=ay+b中所有的数据都在x中传递了,没有其他的参数,所以这里的n_dims就是a和b的维度"""
        super(ODEIVPCase2BSL,self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.t_0 = 0
        self.steps = torch.zeros(1) # 在evaluate中将改为一个大小为(b_size, n_points)的张量，内容对应于每个xs点的步数，每次evaluate(即计算ys)的时候将会更新
    def evaluate(self, xs_b, use_h=False, method = 'RK45'):
        """
        methods = [
            'euler_explicit',  # 显式欧拉方法
            'euler_implicit',  # 隐式欧拉方法
            'euler_improved',  # 改进欧拉方法
            'euler_adaptive',  # 自适应欧拉方法
            'RK45',     # (默认) 显式Runge-Kutta方法 (4,5阶)，适用于非刚性问题
            'RK23',     # 显式Runge-Kutta方法 (2,3阶)，适用于精度要求不高的问题
            'DOP853',   # 高阶显式Runge-Kutta方法 (8阶，13阶精度)，适合高精度需求
            'Radau',    # 隐式Runge-Kutta方法，适用于刚性问题
            'BDF',      # 隐式后向差分公式，适用于刚性问题
            'LSODA'     # 自动切换刚性/非刚性求解器 (基于LSODE算法)
        ]
        """
        # 批量转换参数到CPU numpy数组
        params = xs_b[..., :7].cpu().numpy()  # [batch, points, 7]
        batch_size, num_points, _ = params.shape
        
        all_y_values = []
        
        # 展平批次处理
        for a1, a2, b1, b2, y0, t_e, steps in params.reshape(-1, 7):
            steps = int(steps)
            t = np.linspace(0, t_e, steps)
            
            # ==== 新增求解方法选择逻辑 ====
            if method.startswith('euler'):
                # 显式欧拉
                if method == 'euler_explicit':
                    y_values = [y0]
                    for curr_t in t[1:]:
                        dy = b1 * np.exp(b2 * curr_t) - (a1 * curr_t + a2) * y_values[-1]
                        y_next = y_values[-1] + (t[1]-t[0]) * dy
                        y_values.append(y_next)
                
                # 隐式欧拉
                elif method == 'euler_implicit':
                    y_values = [y0]
                    h = t[1] - t[0]
                    for curr_t in t[1:]:
                        # 解隐式方程 y_{n+1} = y_n + h*(b1*e^{b2 t} - (a1 t + a2)y_{n+1})
                        numerator = y_values[-1] + h * b1 * np.exp(b2 * curr_t)
                        denominator = 1 + h * (a1 * curr_t + a2)
                        y_next = numerator / denominator
                        y_values.append(y_next)
                
                # 改进欧拉
                elif method == 'euler_improved':
                    y_values = [y0]
                    h = t[1] - t[0]
                    for curr_t in t[1:]:
                        # 预测步
                        dy_p = b1 * np.exp(b2 * curr_t) - (a1 * curr_t + a2) * y_values[-1]
                        y_p = y_values[-1] + h * dy_p
                        
                        # 校正步
                        dy_c = b1 * np.exp(b2 * curr_t) - (a1 * curr_t + a2) * y_p
                        y_next = y_values[-1] + h * (dy_p + dy_c) / 2
                        y_values.append(y_next)
                
                else:
                    raise ValueError(f"Unsupported Euler method: {method}")
            
            # 原有方法保持不变
            else:
                def ode_func(t, y):
                    return b1 * np.exp(b2 * t) - (a1 * t + a2) * y
                
                sol = solve_ivp(ode_func, [0, t_e], [y0],
                              method=method, t_eval=t)
                y_values = sol.y[0]
            all_y_values.append(torch.tensor(y_values, device=xs_b.device))

        # 重塑结果形状
        ys_b = torch.zeros(batch_size, num_points, self.n_dims, device=xs_b.device)
        steps_tensor = xs_b[..., 6].long()
        
        for idx, (i, j) in enumerate(np.ndindex(batch_size, num_points)):
            valid_steps = min(steps_tensor[i,j], self.n_dims)
            ys_b[i, j, :valid_steps] = all_y_values[idx][:valid_steps]
            
        return ys_b
        # def ground_truth_tensor(argss):
        #     """
        #     计算变上限积分函数 y(t) 在 [t_0, t_e] 区间内 steps 个均匀分点上的值。
        #     """
        #     [a_1, a_2, b_1, b_2, y_0, t_e, steps] = argss
        #     y_0,steps = y_0.item(),int(steps.item())

        #     # 创建被积函数
        #     def integrand(s):
        #         return b_1 * np.exp((a_1 / 2) * s**2 + (a_2 + b_2) * s)
            
        #     # 创建新的分点用于计算 y(t)
        #     new_t_values = np.linspace(0, t_e, steps)
        #     y_values = []

        #     # 先计算括号中的函数值，使用simps简化计算
        #     inbracket = y_0
        #     y_values.append(inbracket)
        #     pre_t = 0
        #     for t in new_t_values[1:]:
        #         # 总共有steps-1个需要积分计算的区间[pre_t,t]
        #         # 获取 [pre_t,t] 的分点和对应的被积函数值(3个分点 - 11个分点有点恐怖了...300个小时)

        #         t_slice = np.linspace(pre_t, t, 3)
        #         integrand_slice = integrand(t_slice)

        #         # 使用辛普森法则计算五点积分
        #         integral_result = simpson(integrand_slice, x=t_slice)
                
        #         inbracket += integral_result

        #         # 计算 y(t)
        #         y_t = np.exp(-((a_1 / 2) * t**2 + a_2 * t)) * inbracket
        #         y_values.append(y_t)
        #         pre_t = t

            
        #     return torch.tensor(y_values)
        
        # self.steps = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)

        # ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], self.n_dims, device=xs_b.device)
        # for i in range(xs_b.shape[0]):
        #     for j in range(xs_b.shape[1]):
        #         steps = int(xs_b[i, j, 6].item())
        #         ys_b[i, j] = torch.cat([ground_truth_tensor(xs_b[i, j, :7]),torch.zeros(self.n_dims - steps)],dim=0)
        # return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    @staticmethod
    def get_enhanced_metric(step_table):
        # self.steps 
        def error_sq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            return torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
        return error_sq

    @staticmethod
    def get_enhanced_training_metric(step_table):
        def error_msq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            squared_errors = torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
            return squared_errors.mean()
        return error_msq

    @staticmethod
    def get_training_steptable(xs_b):
        # 直接提取步数信息
        return xs_b[..., 6].long()

class ODEIVPCase2vec(Task):
    """
    The problem is 
    dy/dt + p(t)y = q(t),
    p(t) = a_1*t+a_2,
    q(t) = b_1*e^(b_2*t),
    y(t0) = y_0,
    t\in [t_0,t_e],
    要求x格式 x[i,j]=[a_1,a_2,b_1,b_2,y_0,t_e,steps,0,...,0]
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """因为 dy/dt=ay+b中所有的数据都在x中传递了,没有其他的参数,所以这里的n_dims就是a和b的维度"""
        super(ODEIVPCase2vec,self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.t_0 = 0
        self.steps = torch.zeros(1) # 在evaluate中将改为一个大小为(b_size, n_points)的张量，内容对应于每个xs点的步数，每次evaluate(即计算ys)的时候将会更新
    def evaluate(self, xs_b, use_h=False, quadmethod='simps'):
        """
        quadmethod = 'simps'  # 辛普森法则
        quadmethod = 'trapz'  # 梯形法则
        quadmethod = 'rect'   # 矩形法则
        quadmethod = 'romberg'  # 龙贝格积分
        quadmethod = 'quad'  # 自适应积分
        """
        # 将参数一次性转换到CPU并保持张量结构
        params = xs_b[..., :7].cpu().numpy()  # shape [batch, points, 7]
        batch_size, num_points, _ = params.shape
        
        # 向量化参数处理
        a1, a2, b1, b2, y0, te, steps = params[...,0], params[...,1], params[...,2], params[...,3], params[...,4], params[...,5], params[...,6]
        
        steps = steps.astype(int)

        # 修复时间网格生成逻辑
        max_steps = steps.max()
        t_all = np.zeros((batch_size, num_points, max_steps))
        for i in range(batch_size):
            for j in range(num_points):
                step_val = int(steps[i,j])
                step_val = max(1, step_val)  # 至少1个时间点
                t_all[i,j,:step_val] = np.linspace(0, te[i,j], num=step_val, endpoint=True)
                t_all[i,j,step_val:] = te[i,j]

        # 同时修改后续的断言检查
        for i in range(batch_size):
            for j in range(num_points):
                step_val = int(steps[i,j])
                # 检查最后一个有效点是否等于te
                if not np.isclose(t_all[i,j,step_val-1], te[i,j], atol=1e-6):
                    raise ValueError(f"时间网格生成错误: {t_all[i,j,step_val-1]} vs {te[i,j]}")

        # 方法选择器
        if quadmethod == 'simps':
            # 辛普森法则（当前默认方法）
            mid_points = (t_all[..., :-1] + t_all[..., 1:]) / 2
            exp_part_start = (a1[..., None]/2)*(t_all[..., :-1]**2) + (a2[..., None] + b2[..., None])*t_all[..., :-1]
            exp_part_mid = (a1[..., None]/2)*(mid_points**2) + (a2[..., None] + b2[..., None])*mid_points
            exp_part_end = (a1[..., None]/2)*(t_all[..., 1:]**2) + (a2[..., None] + b2[..., None])*t_all[..., 1:]
            dt = np.diff(t_all, axis=2)
            integrals = b1[..., None] * (
                np.exp(exp_part_start) + 
                4 * np.exp(exp_part_mid) + 
                np.exp(exp_part_end)
            ) * dt / 6
            
        elif quadmethod == 'trapz':
            # 梯形法则（原注释代码）
            mid_points = (t_all[..., :-1] + t_all[..., 1:]) / 2
            exp_part_start = (a1[..., None]/2)*(t_all[..., :-1]**2) + (a2[..., None] + b2[..., None])*t_all[..., :-1]
            exp_part_mid = (a1[..., None]/2)*(mid_points**2) + (a2[..., None] + b2[..., None])*mid_points
            dt = np.diff(t_all, axis=2)
            integrals = b1[..., None] * (np.exp(exp_part_start) + np.exp(exp_part_mid)) / 2 * dt
            
        elif quadmethod == 'rect':
            # 矩形法则（原注释代码）
            dt = np.diff(t_all, axis=2)
            exp_part = (a1[..., None]/2)*(t_all[..., :-1]**2) + (a2[..., None] + b2[..., None])*t_all[..., :-1]
            integrals = b1[..., None] * np.exp(exp_part) * dt
        
        elif quadmethod == 'romberg':
            # 龙贝格积分（误差阶 O(h^2m)）
            # 计算四个细分点（t1/4, t1/2, t3/4, t1）
            t0 = t_all[..., :-1]
            t1 = t_all[..., 1:]
            
            # 生成四个中间点
            t_14 = t0 + (t1 - t0)*0.25
            t_12 = (t0 + t1)/2
            t_34 = t0 + (t1 - t0)*0.75
            
            # 计算五个点的被积函数值
            f0 = np.exp((a1[..., None]/2)*t0**2 + (a2[..., None]+b2[..., None])*t0)
            f1 = np.exp((a1[..., None]/2)*t_14**2 + (a2[..., None]+b2[..., None])*t_14)
            f2 = np.exp((a1[..., None]/2)*t_12**2 + (a2[..., None]+b2[..., None])*t_12)
            f3 = np.exp((a1[..., None]/2)*t_34**2 + (a2[..., None]+b2[..., None])*t_34)
            f4 = np.exp((a1[..., None]/2)*t1**2 + (a2[..., None]+b2[..., None])*t1)
            
            # 龙贝格递推公式
            integrals = b1[..., None] * (t1 - t0) * (7*f0 + 32*f1 + 12*f2 + 32*f3 + 7*f4) / 90

        elif quadmethod == 'adaptive':
            # 初始化时间区间变量
            t0 = t_all[..., :-1]  # 形状 [batch, points, max_steps]
            t1 = t_all[..., 1:]   # 形状 [batch, points, max_steps]
            
            # 生成高斯积分点（优化维度扩展方式）
            t_gauss = t0[..., None] * 0.9530899 + t1[..., None] * 0.0469101  # 使用更高效的广播方式
            weights = np.array([0.1184635, 0.2393144, 0.2844444, 0.2393144, 0.1184635])
            
            # 计算高斯点处的被积函数（优化维度对齐）
            exponents = (
                (a1[..., None, None]/2) * t_gauss**2 + 
                (a2[..., None, None] + b2[..., None, None]) * t_gauss
            )
            f_gauss = np.exp(exponents)
            
            # 重新调整维度进行加权求和
            integral_components = np.einsum('i,...i->...i', weights, f_gauss)  # 保持最后一个维度
            time_intervals = t1 - t0  # 形状 [batch, points, max_steps]
            
            # 最终积分计算（调整维度广播）
            integrals = (
                b1[..., None, None] *          # 形状 [batch, points, 1, 1]
                time_intervals[..., None] *     # 形状 [batch, points, max_steps, 1]
                integral_components             # 形状 [batch, points, max_steps, 5]
            ).sum(axis=-1)  # 沿最后一个维度求和，最终形状 [batch, points, max_steps]
        else:
            raise ValueError(f"Unsupported integration quadmethod: {quadmethod}")

        # 恢复向量化实现并修正衰减因子
        cumulative_integrals = np.cumsum(integrals, axis=2) #累计求和分段实现d
        
        # 修正初始项计算
        initial_term = y0[..., None] * np.exp(
            (a1[..., None]/2)*t_all[..., 0, None]**2 +
            a2[..., None]*t_all[..., 0, None]
        )
        
        
        # 最终计算时包含初始值
        y_values = np.zeros((batch_size, num_points, max_steps))  # 初始化全零数组
        y_values[..., 0] = y0  # 设置初始值

        # 计算后续时间点的值
        decay_t = t_all[..., 1:]
        decay_factor = np.exp(
            -(a1[..., None]/2)*(decay_t**2) -
            a2[..., None]*decay_t
        )
        y_values[..., 1:] = decay_factor * (initial_term + cumulative_integrals)

        # 新增稳定性处理
        decay_factor = np.clip(decay_factor, 1e-12, 1e12)
        cumulative_integrals = np.clip(cumulative_integrals, -1e24, 1e24)
        # 转换为PyTorch张量并填充结果
        ys_b = torch.zeros(batch_size, num_points, self.n_dims, device=xs_b.device)
        for i in range(batch_size):
            for j in range(num_points):
                valid_steps = min(steps[i,j], self.n_dims)
                # 保持原状，现在维度已对齐
                ys_b[i,j,:valid_steps] = torch.from_numpy(y_values[i,j,:valid_steps])

        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    @staticmethod
    def get_enhanced_metric(step_table):
        # self.steps 
        def error_sq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            return torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
        return error_sq

    @staticmethod
    def get_enhanced_training_metric(step_table):
        def error_msq(ys, ys_true):
            # ys: (b_size, n_points, n_dims)
            # ys_true: (b_size, n_points, n_dims)
            # 使用掩码操作替代循环
            mask = torch.arange(ys.shape[-1], device=ys.device).expand(ys.shape[0], ys.shape[1], -1) < step_table.unsqueeze(-1)
            squared_errors = torch.where(mask, (ys - ys_true).square(), torch.zeros_like(ys))
            return squared_errors.mean()
        return error_msq

    @staticmethod
    def get_training_steptable(xs_b,use_h = False):
        # 直接提取步数信息
        if use_h:
            return ((xs_b[..., 5] / xs_b[..., 6]) + 1).long()
        return xs_b[..., 6].long()
