import os
from random import randint
import uuid
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import numpy as np
from tasks import get_task_sampler
from samplers import get_data_sampler
import random
import wandb

def generate_dataset(
    n_points,
    taskname,
    dataname,
    n_dims,
    bsize,
    seed,
    step_nums,
    save_path,
):
    data_sampler = get_data_sampler(dataname,n_dims=n_dims)
    task_sampler_func = get_task_sampler(taskname,n_dims=n_dims,batch_size=bsize)
    task_sampler_args = {}
    task_sampler = task_sampler_func(**task_sampler_args)
    pbar = tqdm(range(step_nums),desc='generate dataset')
    for i in pbar:
        xs = data_sampler.sample_xs(
            n_points,
            bsize
        )
        ys = task_sampler.evaluate(xs)