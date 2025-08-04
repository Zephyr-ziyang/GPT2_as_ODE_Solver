from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge
# 配置管理和参数校验 json

model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm","gptJ"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
}

curriculum_base_schema = { #
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

lr_schema = {
    "start": merge(tfloat, required),        # 初始学习率
    "steadylearn": merge(tfloat, required),  # 稳定期学习率
    "schedule_type": merge(tstring, allowed(["stable", "four_stage"]), default("stable")),  # 默认stable
    
    # 四阶段专用参数设为可选
    "firstup_end": merge(tinteger, nullable, default(None)), 
    "highlearn": merge(tfloat, nullable, default(None)),
    "highlearn_end": merge(tinteger, nullable, default(None)),
    "downlearn_end": merge(tinteger, nullable, default(None)),
    "lowlearn": merge(tfloat, nullable, default(None))
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
    "lr": stdict(lr_schema)  # 新增学习率调度配置
}

TASK_LIST = [ # add tasks
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
    "ode_ivp_case1",
    "ode_ivp_case1plus",
    "ode_ivp_case2",
    "ode_ivp_case2vec",
    "ode_ivp_case2plus",

    # "linear_regression_uniform",
]

TASK_TO_DATA = {
    "linear_regression": "gaussian",
    "ode_ivp_case1": "ode_ivp_case1",
    "ode_ivp_case1plus": "ode_ivp_case1plus",
    "ode_ivp_case2": "ode_ivp_case2",
    "ode_ivp_case2vec": "ode_ivp_case2",
    "ode_ivp_case2plus": "ode_ivp_case2",

}
DATA_LIST = [
    "gaussian",
    "uniform",
    "ode_ivp_case1",
    "ode_ivp_case1plus",
    "ode_ivp_case2",
    "ode_ivp_case2plus",
]
training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(DATA_LIST)), # data type
    "w_type":merge(tstring,allowed(["gaussian","uniform"]),default("gaussian")), # task w distribution
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(1e-4)),  #  large model 调大
    "weight_decay": merge(tfloat, default(0.00)), # for Adam
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
    "If_two_distribution": merge(tboolean, nullable, default(False)), # data 一半来自于distribution1 ， 一半 2
    "If_RandomShuffle_2distribution": merge(tboolean, nullable, default(False)),# 两类distribution 的batch打乱
    "w_distribution1": merge(tstring, allowed(["gaussian","uniform"]),default("gaussian")),
    "w_distribution2": merge(tstring, allowed(["gaussian", "uniform"]), default("uniform")),
    "If_enhanced_loss": merge(tboolean, nullable, default(False)),
    "If_use_h": merge(tboolean, nullable, default(False)),
    "If_evalloss_in_train": merge(tboolean, nullable, default(False)),
    "eval_loss_per_steps": merge(tinteger, nullable, default(100)),
    "num_eval_tasks": merge(tinteger, nullable, default(200)),
}
eval_schema = {
    # "evalmode": merge(tboolean, nullable, default(False)),
    "If_shift_w_distribution": merge(tboolean, nullable, default(False)),
    "eval_w_type": merge(tstring, allowed(["add"]), default("add")),  # w1+w2
    # "task": merge(tstring, allowed(TASK_LIST),default("notallowed_task")),
    # "data": merge(tstring, allowed(DATA_LIST),default("notallowed_data")),
    # "n_dims": merge(tinteger, default(20)),
    # "n_points": merge(tinteger, default(41)),
    # "batch_size": merge(tinteger, default(64)),
}
wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "eval": stdict(eval_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)), #
    "trained_id": merge(tstring, nullable, default(None)), # 训练后的模型id
    "set_device": merge(tinteger,allowed([0,1,2,3]), default(0)),
}
