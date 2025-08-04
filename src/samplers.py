import math

import torch
import random
import numpy as np

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError

def rand_select_sampler(sampler1, sampler2): # todo 添加随机选择sample
    if random.random() < 0.5:
        return sampler1
    else:
        return sampler2


def get_data_sampler(data_name, n_dims, **kwargs):
    """
    data_name:数据格式类别
            允许:'gassian','uniform','ode_ivp_case1'
            
    """
    names_to_classes = {
        "ode_ivp_case1": ODEIVPCase1Sampler,
        "ode_ivp_case1plus": ODEIVPCase1plusSampler,
        "ode_ivp_case2": ODEIVPCase2Sampler,
        # add
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        if data_name == "ode_ivp_case1":
            assert n_dims > 4
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError



def sample_transformation(eigenvalues, normalize=False): # 根据给定的特征值（eigenvalues）生成一个线性变换矩阵，用于对数据进行线性变换
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class ODEIVPCase1Sampler(DataSampler):
    """
    dy/dt = ay+b,
    y(t0) = t0,
    t\in [t0,t1],
    生成问题格式 x:[a,b,y0,steps,0,0,0,...,0]
    注: steps = 要求生成的y的分点数量,即步长为t_e-t0/(steps-1)
    暂时fix:
        t_0=0
    随机生成设置
        a_st,a_ed = 1,2
        b_st,b_ed = -2,2
        y0_st,y0_ed = -1,1
    """
    def __init__(self, n_dims, scale=None,bias=None):
        super().__init__(n_dims)
        self.scale = scale
        self.bias = bias
    def sample_xs(self, n_points, b_size, use_h=False):  # 添加use_h参数
        assert self.n_dims >= 5 # 至少5个维度
        # 随机生成设置
        a_st,a_ed = 1,2
        b_st,b_ed = -2,2
        y0_st,y0_ed = -1,1
        t_e_st,t_e_ed = 0,1
        step_st,step_ed = 5,self.n_dims
        h_short, h_long = t_e_ed/step_ed, t_e_ed/step_st
        # if seeds is None:

        # 随机生成a
        a = a_st + (a_ed - a_st) * torch.rand(b_size,1)
        a = a.expand(-1, n_points)
        # 随机生成b
        b = b_st + (b_ed - b_st) * torch.rand(b_size,1)
        b = b.expand(-1, n_points)
        # 随机生成y0
        y0 = y0_st + (y0_ed - y0_st) * torch.rand(b_size,1)
        y0 = y0.expand(-1, n_points)

        # 修改steps生成逻辑
        if use_h:
            h = h_short + (h_long - h_short) * torch.rand(b_size, n_points, device=self.device)
            testrange = h
        else:
            steps = torch.randint(step_st, step_ed, (b_size, n_points), device=self.device)
            testrange = steps
        
        x_s = torch.stack([a, b, y0, testrange], dim=2)

        # else :
        #     x_s = torch.zeros(b_size, n_points, n_dims)
        #     generator = torch.Generator()
        #     assert len(seeds) == b_size
        #     for i, seed in enumerate(seeds):
        #         generator.manual_seed(seed)
        #         # 生成a、b、y0
        #         a_b_y0 = torch.rand(3, generator=generator)

        #         # 缩放
        #         a = a_st + (a_ed - a_st) * torch.tensor([a_b_y0[0]])
        #         a = a.expand(-1, n_points)
        #         b = b_st + (b_ed - b_st) * torch.tensor([a_b_y0[1]])
        #         b = b.expand(-1, n_points)
        #         y0 = y0_st + (y0_ed - y0_st) * torch.tensor([a_b_y0[2]])
        #         y0 = y0.expand(-1, n_points)

        #         steps = torch.randint(step_st, step_ed, (n_points))
        #         x_s[i] = torch.stack([a, b, y0, steps], dim=2)

        if self.scale is not None:
            xs_b = xs_b @ self.scale    
        if self.bias is not None:
            xs_b += self.bias

        # 做dim fix，补零
        zeros = torch.zeros(b_size, n_points, self.n_dims - 4)
        x_s = torch.cat([x_s, zeros], dim=2)
        return x_s
    def continuous_sample_xs(self, n_points, b_size, use_h = False):
        # 随机生成设置
        a_st,a_ed = 1,2
        b_st,b_ed = -2,2
        y0_st,y0_ed = -1,1
        step_st,step_ed = 5,self.n_dims

        # 随机生成a
        a = a_st + (a_ed - a_st) * torch.rand(b_size,1)
        a = a.expand(-1, n_points)
        # 随机生成b
        b = b_st + (b_ed - b_st) * torch.rand(b_size,1)
        b = b.expand(-1, n_points)
        # 随机生成y0
        y0 = y0_st + (y0_ed - y0_st) * torch.rand(b_size,1)
        y0 = y0.expand(-1, n_points)
        # # 固定生成a
        # a = torch.full((b_size, n_points), a_st)
        # # 固定生成b
        # b = torch.full((b_size, n_points), b_st)
        # # 固定生成y0
        # y0 = torch.full((b_size, n_points), y0_st)
        # 从小到大生成steps,剩余补为中间值
        # 从小到大,每一次加一生成steps,剩余补为中间值
        steps = torch.zeros(b_size, n_points)
        for i in range(b_size):
            if (step_ed - step_st) < n_points:
                steps[i, :step_ed - step_st] = torch.arange(step_st, step_ed, device=self.device)
                steps[i, step_ed - step_st:] = int((step_st + step_ed) / 2)
            else:
                steps[i, :] = torch.arange(step_st, step_st+n_points, device=self.device)

        x_s = torch.stack([a, b, y0, steps], dim=2)

        # 做dim fix，补零
        zeros = torch.zeros(b_size, n_points, self.n_dims - 4)
        x_s = torch.cat([x_s, zeros], dim=2)
        return x_s

class ODEIVPCase1plusSampler(DataSampler):
    """
    dy/dt = ay+b,
    y(t0) = y0,
    t\in [t0,t1],
    生成问题格式 x:[a,b,y0,t_e,steps,0,0,0,...,0]
    注: steps = 要求生成的y的分点数量,即步长为t1-t0/(steps-1)
    暂时fix:
        t_0=0
    随机生成设置
        a_st,a_ed = 1,2
        b_st,b_ed = -2,2
        y0_st,y0_ed = -1,1
        t_e_st,t_e_ed = 1,2
    """
    def __init__(self, n_dims, scale=None,bias=None):
        super().__init__(n_dims)
        self.scale = scale.to('cuda') if scale is not None and torch.cuda.is_available() else scale
        self.bias = bias.to('cuda') if bias is not None and torch.cuda.is_available() else bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_xs(self, n_points, b_size, n_dims_truncated = None, use_h = False,**kwags):
        assert self.n_dims >= 5 # 至少5个维度
        # 随机生成设置
        a_st,a_ed = 1,2
        b_st,b_ed = -2,2
        y0_st,y0_ed = -1,1
        step_st,step_ed = 5,self.n_dims-5
        t_e_st,t_e_ed = 1,2
        # if seeds is None:

        # 随机生成a
        a = a_st + (a_ed - a_st) * torch.rand(b_size,1, device=self.device)
        a = a.expand(-1, n_points)
        # 随机生成b
        b = b_st + (b_ed - b_st) * torch.rand(b_size,1, device=self.device)
        b = b.expand(-1, n_points)
        # 随机生成y0
        y0 = y0_st + (y0_ed - y0_st) * torch.rand(b_size,1, device=self.device)
        y0 = y0.expand(-1, n_points)
        
        t_e = t_e_st + (t_e_ed - t_e_st) * torch.rand(b_size,1, device=self.device)
        t_e = t_e.expand(-1, n_points)
        
        # 检测是h步长模式还是steps步数模式
        if use_h == True:
            # 随机生成h
            h_short, h_long = t_e_ed/step_ed, t_e_st/step_st
            h = h_short + (h_long-h_short) * torch.rand(b_size, n_points, device=self.device)
            testrange = h
        else:
            # 随机生成steps
            steps = torch.randint(step_st, step_ed, (b_size, n_points), device=self.device)
            testrange = steps

        x_s = torch.stack([a, b, y0, t_e, testrange], dim=2)

        # 做dim fix，补零
        zeros = torch.zeros(b_size, n_points, self.n_dims - 5, device=self.device)
        x_s = torch.cat([x_s, zeros], dim=2)
        return x_s

    def continuous_sample_xs(self, n_points, b_size, use_h=False,**kwags):
        """生成连续样本数据，每个batch共享a/b/t_e参数，支持两种模式"""
        assert self.n_dims >= 5
        
        # 参数生成范围设置
        a_st,a_ed = 1,2
        b_st,b_ed = -2,2
        y0_st,y0_ed = -1,1
        step_st,step_ed = 5,self.n_dims-5
        t_e_st,t_e_ed = 1,2

        # 生成共享参数
        a_val = a_st + (a_ed - a_st) * torch.rand(1, device=self.device).item()
        b_val = b_st + (b_ed - b_st) * torch.rand(1, device=self.device).item()
        t_e_val = t_e_st + (t_e_ed - t_e_st) * torch.rand(1, device=self.device).item()

        # 扩展共享参数
        a = torch.full((b_size, n_points), a_val, device=self.device)
        b = torch.full((b_size, n_points), b_val, device=self.device)
        t_e = torch.full((b_size, n_points), t_e_val, device=self.device)

        # 生成不同的初始值y0
        y0 = y0_st + (y0_ed - y0_st) * torch.rand(b_size, 1, device=self.device)
        y0 = y0.expand(-1, n_points)

        # 动态生成steps矩阵（与步数模式相同）
        # steps = torch.zeros(b_size, n_points, device=self.device)
        # for i in range(b_size):
        #     current_step_ed = self.n_dims - (n_points - i)
        #     current_step_ed = max(current_step_ed, step_st)
        #     current_step_ed = step_ed
            
        #     valid_steps = torch.arange(step_st, current_step_ed + 1, device=self.device)
        #     if len(valid_steps) >= n_points:
        #         steps[i] = valid_steps[:n_points]
        #     else:
        #         last_val = valid_steps[-1].repeat(n_points - len(valid_steps))
        #         steps[i] = torch.cat([valid_steps, last_val])
        
        steps = torch.zeros(b_size, n_points, device=self.device)
        for i in range(b_size):
            steps[i] = torch.linspace(step_st, step_ed, n_points, device=self.device)


        # 模式选择
        if use_h:
            # 步长模式
            # 计算步长为 t_e / steps
            h = t_e_val / steps
            testrange = h
        else:
            # 步数模式
            testrange = steps

        # 拼接参数
        x_s = torch.stack([a, b, y0, t_e, testrange], dim=2)

        # 维度补齐
        zeros = torch.zeros(b_size, n_points, self.n_dims - 5, device=self.device)
        x_s = torch.cat([x_s, zeros], dim=2)
        return x_s

class ODEIVPCase2Sampler(DataSampler):
    """
    dy/dt + p(t)y = q(t),
    p(t) = a_1*t+a_2,
    q(t) = b_1*e^(b_2*t),
    y(t0) = y_0,
    t\in [t_0,t_e],
    生成问题格式 x:[a_1,a_2,b_1,b_2,y_0,t_e,steps,0,...,0]
    注: steps = 要求生成的y的分点数量,即步长为t_e-t_0/(steps-1)
    暂时fix:
        t_0=0
    随机生成设置
        a_1_st,a_1_ed = -1,1
        a_2_st,a_2_ed = -2,2
        b_1_st,b_1_ed = -2,2
        b_2_st,b_2_ed = -3,3
        y_0_st,y_0_ed = -1,1
        t_e_st,t_e_ed = 0,2
        steps_st,steps_ed = 5,n_dims
    """
    def __init__(self, n_dims, scale=None,bias=None):
        super().__init__(n_dims)
        # 将 scale 和 bias 移动到 GPU 上
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale.to(self.device) if scale is not None else scale
        self.bias = bias.to(self.device) if bias is not None else bias

    def sample_xs(self, n_points, b_size, use_h = False, shifted_paras = None, **kwags):
        """
        参数：
        n_points: 每个样本的点数量
        b_size: 批次大小
        可选：
        use_h: 是否使用 h 模式(将返回steps的位置修改为h)
        shifted_paras: 用于固定参数的字典，格式为：
            {
                'para_k': [start, end],
                ...
            }
        """
        assert self.n_dims >= 7 # 至少7个维度

        if shifted_paras is not None:
            # 从 shifted_paras 中提取参数
            a_1_st, a_1_ed = shifted_paras['a_1']
            a_2_st, a_2_ed = shifted_paras['a_2']
            b_1_st, b_1_ed = shifted_paras['b_1']
            b_2_st, b_2_ed = shifted_paras['b_2']
            y_0_st, y_0_ed = shifted_paras['y_0']
            t_e_st, t_e_ed = shifted_paras['t_e']
            step_st, step_ed = shifted_paras['steps']
        else:
            # 随机生成设置
            a_1_st,a_1_ed = -1,1
            a_2_st,a_2_ed = -2,2
            b_1_st,b_1_ed = -2,2
            b_2_st,b_2_ed = -3,3
            y_0_st,y_0_ed = -1,1
            t_e_st,t_e_ed = 1,2
            step_st,step_ed = 5,self.n_dims
        # if seeds is None:

        # 随机生成，将张量创建在 GPU 上
        a_1 = a_1_st + (a_1_ed - a_1_st) * torch.rand(b_size, 1, device=self.device)
        a_1 = a_1.expand(-1, n_points)
        a_2 = a_2_st + (a_2_ed - a_2_st) * torch.rand(b_size, 1, device=self.device)
        a_2 = a_2.expand(-1, n_points)
        b_1 = b_1_st + (b_1_ed - b_1_st) * torch.rand(b_size, 1, device=self.device)
        b_1 = b_1.expand(-1, n_points)
        b_2 = b_2_st + (b_2_ed - b_2_st) * torch.rand(b_size, 1, device=self.device)
        b_2 = b_2.expand(-1, n_points)
        y_0 = y_0_st + (y_0_ed - y_0_st) * torch.rand(b_size, 1, device=self.device)
        y_0 = y_0.expand(-1, n_points)
        if use_h:
            h_short, h_long = t_e_ed/step_ed, t_e_st/step_st
            h = h_short + (h_long - h_short) * torch.rand(b_size, n_points, device=self.device)
            testrange = h
        else:
            steps = torch.randint(step_st, step_ed, (b_size, n_points), device=self.device)
            testrange = steps
        t_e = t_e_st + (t_e_ed - t_e_st) * torch.rand(b_size, 1, device=self.device)
        t_e = t_e.expand(-1, n_points)

        x_s = torch.stack([a_1, a_2, b_1, b_2, y_0, t_e, testrange], dim=2)

        # 做dim fix，补零，将零张量创建在 GPU 上
        zeros = torch.zeros(b_size, n_points, self.n_dims - 7, device=self.device)
        x_s = torch.cat([x_s, zeros], dim=2)
        return x_s

    def continuous_sample_xs(self, n_points, b_size, use_h = False, shifted_paras = None, **kwags):
        """生成连续样本数据，每个batch共享a1/a2/b1/b2/t_e参数，仅y0不同"""
        assert self.n_dims >= 7  # 确保输入维度足够（至少包含7个参数）
        
        if shifted_paras is not None:
            # 从 shifted_paras 中提取参数
            a_1_st, a_1_ed = shifted_paras['a_1']
            a_2_st, a_2_ed = shifted_paras['a_2']
            b_1_st, b_1_ed = shifted_paras['b_1']
            b_2_st, b_2_ed = shifted_paras['b_2']
            y_0_st, y_0_ed = shifted_paras['y_0']
            t_e_st, t_e_ed = shifted_paras['t_e']
            step_st, step_ed = shifted_paras['steps']
        else:
            # 随机生成设置
            a_1_st,a_1_ed = -1,1
            a_2_st,a_2_ed = -2,2
            b_1_st,b_1_ed = -2,2
            b_2_st,b_2_ed = -3,3
            y_0_st,y_0_ed = -1,1
            t_e_st,t_e_ed = 1,2
            step_st,step_ed = 5,self.n_dims-5

        # 生成共享参数（整个batch使用相同的参数）
        a_1_val = a_1_st + (a_1_ed - a_1_st) * torch.rand(1, device=self.device).item()
        a_2_val = a_2_st + (a_2_ed - a_2_st) * torch.rand(1, device=self.device).item()
        b_1_val = b_1_st + (b_1_ed - b_1_st) * torch.rand(1, device=self.device).item()
        b_2_val = b_2_st + (b_2_ed - b_2_st) * torch.rand(1, device=self.device).item()
        t_e_val = t_e_st + (t_e_ed - t_e_st) * torch.rand(1, device=self.device).item()

        # 扩展共享参数到整个batch
        a_1 = torch.full((b_size, n_points), a_1_val, device=self.device)
        a_2 = torch.full((b_size, n_points), a_2_val, device=self.device)
        b_1 = torch.full((b_size, n_points), b_1_val, device=self.device)
        b_2 = torch.full((b_size, n_points), b_2_val, device=self.device)
        t_e = torch.full((b_size, n_points), t_e_val, device=self.device)

        # 每个prompt生成不同的初始值y0
        y_0 = y_0_st + (y_0_ed - y_0_st) * torch.rand(b_size, 1, device=self.device)
        y_0 = y_0.expand(-1, n_points)  # 扩展到所有时间点

        steps = torch.zeros(b_size, n_points, device=self.device)
        for i in range(b_size):
            current_step_ed = self.n_dims - (n_points - i)
            current_step_ed = max(current_step_ed, step_st)
            
            valid_steps = torch.arange(step_st, current_step_ed + 1, device=self.device)
            if len(valid_steps) >= n_points:
                steps[i] = valid_steps[:n_points]
            else:
                last_val = valid_steps[-1].repeat(n_points - len(valid_steps))
                steps[i] = torch.cat([valid_steps, last_val])
        
        # steps = torch.zeros(b_size, n_points, device=self.device)
        # for i in range(b_size):
        #     steps[i] = torch.linspace(step_st, step_ed, n_points, device=self.device)

        # 模式选择逻辑
        if use_h:
            # 步长模式
            h = t_e_val / steps
            testrange = h
        else:
            # 步数模式
            testrange = steps

        # 拼接所有参数形成样本
        x_s = torch.stack([a_1, a_2, b_1, b_2, y_0, t_e, testrange], dim=2)

        # 补零操作保证维度统一
        zeros = torch.zeros(b_size, n_points, self.n_dims - 7, device=self.device)
        x_s = torch.cat([x_s, zeros], dim=2)
        return x_s