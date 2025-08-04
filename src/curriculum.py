import math


class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        #
        # 初始化课程学习参数
        self.n_dims_truncated = args.dims.start  # 当前使用的特征维度数
        self.n_points = args.points.start       # 当前使用的数据点数量
        self.n_dims_schedule = args.dims        # 维度变化策略配置
        self.n_points_schedule = args.points    # 数据量变化策略配置
        self.step_count = 0                     # 训练步数计数器

        self.current_lr = args.lr.start  # 新增学习率参数
        self.lr_schedule = args.lr       # 学习率调度配置


    def update(self):
        # 每训练step完成时调用，更新参数
        self.step_count += 1
        self.n_dims_truncated = self.update_var(  # 更新特征维度
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(  # 更新数据量
            self.n_points, self.n_points_schedule
        )
        self.current_lr = self.update_lr()  # 新增学习率更新

    def update_lr(self):
        # 学习率调度策略

        """多种策略选择：
        1.稳定学习率调度策略
        2.四阶段混合调度策略
        """
        if self.lr_schedule.schedule_type == "stable":
            return self.lr_schedule.steadylearn
        elif self.lr_schedule.schedule_type == "four_stage":
        # 阶段1：线性上升期
            if self.step_count <= self.lr_schedule.firstup_end:
                progress = self.step_count / self.lr_schedule.firstup_end
                return self.lr_schedule.start + progress * (
                    self.lr_schedule.highlearn - self.lr_schedule.start
                )
            
            # 阶段2：高学习率平台期
            if self.step_count <= self.lr_schedule.highlearn_end:
                return self.lr_schedule.highlearn
            
            # 阶段3：余弦下降期
            if self.step_count <= self.lr_schedule.downlearn_end:
                decay_steps = self.lr_schedule.downlearn_end - self.lr_schedule.highlearn_end
                current_step = self.step_count - self.lr_schedule.highlearn_end
                ratio = min(current_step / decay_steps, 1.0)
                decay = (1 + math.cos(math.pi * ratio)) / 2  # 余弦衰减因子
                return self.lr_schedule.lowlearn + (
                    self.lr_schedule.highlearn - self.lr_schedule.lowlearn
                ) * decay
            
            # 阶段4：稳定期
            return self.lr_schedule.steadylearn
    
    def update_var(self, var, schedule):
        # 核心更新逻辑：每隔 interval 步增加 inc
        if self.step_count % schedule.interval == 0:
            var += schedule.inc  # 按增量调整参数
        return min(var, schedule.end)  # 不超过预设的最大值


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)
