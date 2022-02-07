import argparse
import math
import time
import multiprocessing as mp
from datetime import datetime
from typing import List
from pathlib import Path

from renderer import *

import torch
import torchvision.transforms as transforms
from torch.nn import functional as F

import numpy as np
from tqdm import trange
from pgpelib import PGPE
from termcolor import cprint

import clip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--device', type=str)
    parser.add_argument('--n_population', type=int, default=128)
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--n_primitives', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--coordinate_scale', type=float, default=1.0)
    parser.add_argument('--scale_max', type=float, default=1.0)
    parser.add_argument('--scale_min', type=float, default=0.01)
    parser.add_argument('--renderer', type=str, default='BoxRenderer')
    parser.add_argument('--loss_type', type=str, default='cosine')

    args = parser.parse_args()
    return args

def process_augment_renders(renders: List[np.ndarray], device: str, num_augs=4):
    t = np.stack(renders, axis=0).transpose(0, 3, 1, 2)
    t = torch.tensor(t).to(device)
    t = t.type(torch.float32)
    t = t.repeat_interleave(num_augs, dim=0)
    new_augment_trans = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return new_augment_trans(t)

def fitness(batch, text_features, model, num_renders, num_augs=4, loss_type='cosine'):
    with torch.no_grad():
        image_features = model.encode_image(batch)
        if loss_type == 'cosine':
            fit = torch.cosine_similarity(image_features, text_features, axis=-1)
        elif loss_type == 'spherical_dist_loss':
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            # Make negative since we're maximizing.
            fit = -(image_features - text_features).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

        fit = torch.reshape(fit, (num_renders, num_augs)).mean(axis=-1)
    
    fit = fit.to('cpu').tolist()
    return fit

def get_renderer_args(args):
    renderer_args = {
        'n_primitives': args.n_primitives,
        'width': 256,
        'height': 256,
        'coordinate_scale': args.coordinate_scale,
        'scale_max': args.scale_max,
        'scale_min': args.scale_min,
        'random_rotate': False
    }
    return renderer_args

def get_renderer_class(name):
    renderer = __import__('renderer')
    return getattr(renderer, name)

# For multiprocessing pool.
worker_renderer = None
def init_worker(args):
    global worker_renderer
    renderer_cls = get_renderer_class(args.renderer)
    renderer_args = get_renderer_args(args)
    worker_renderer = renderer_cls(**renderer_args)

def render_fn(params):
    global worker_renderer
    render = worker_renderer.render(params)
    return render


def training_loop(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize Renderer
    renderer_cls = get_renderer_class(args.renderer)
    renderer_args = get_renderer_args(args)
    renderer = renderer_cls(**renderer_args)

    # Initialize PGPE Solver
    solver = PGPE(
        solution_length=renderer.n_params*renderer.n_primitives,
        popsize=args.n_population,
        optimizer='clipup',
        optimizer_config={'max_speed': 0.2},
        seed=args.seed
    )

    # Initialize CLIP model + Prompt embedding
    clip_model = clip.load('ViT-B/16', jit=True, device=device)[0]
    text_input = clip.tokenize(args.prompt).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input)

    with Path("output/fitnesses.txt").open('w+') as f:
        f.truncate(0)

    render_pool = mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(args,))

    # Evolutionary loop
    n_iterations = args.n_iterations
    n_augs = 4
    for i in trange(n_iterations):
        try:
            solutions = solver.ask()
            solutions = [s.reshape((renderer.n_primitives, renderer.n_params)) for s in solutions]
            
            # Render each solution from the solver
            t1 = time.time()
            renders = render_pool.map(func=render_fn, iterable=solutions)
            renders = [render for render in renders]
            t2 = time.time()
            print(f"Rendering time: {(t2 - t1):.4f}s")

            # Process and augment all renders
            t1 = time.time()
            im_batch = [process_augment_renders(r, device) for r in renders]
            t2 = time.time()
            print(f"Processing / Augmenting time: {(t2 - t1):.4f}s")

            # Rearrange im_batch into even batch_size chunks
            n_chunks = math.ceil((args.n_population * n_augs) / args.batch_size)

            im_batch = torch.stack(im_batch, dim=0).reshape(-1, 3, 224, 224)
            im_batches = torch.chunk(im_batch, n_chunks) # n_chunks x [batch, 3, 224, 224]

            chunk_size = args.n_population // n_chunks    

            t1 = time.time()    
            fitnesses = [fitness(batch, text_features, clip_model, num_renders=chunk_size, num_augs=n_augs, loss_type=args.loss_type) for batch in im_batches]
            t2 = time.time()
            print(f"CLIP Fitness calculation time: {(t2 - t1):.4f}s")
            fitnesses = np.concatenate(fitnesses)

            t1 = time.time()
            solver.tell(fitnesses)
            t2 = time.time()
            print(f"Solver time: {(t2 - t1):.4f}s")

            # TODO This can be replaced by hooks.
            if i % 20 == 0:
                best_solution = solver.center.reshape(renderer.n_primitives, renderer.n_params)
                renderer.render(best_solution, save_image=f"./output/frame_{i:04}.jpg")
                cprint(f"Fitness: {np.max(fitnesses):.8f}", "green")
                with Path("output/fitnesses.txt").open('a') as f:
                    f.write(f"[{datetime.now()}] Iteration: {i} Fitness: {np.max(fitnesses):.8f}\n")
        except KeyboardInterrupt:
            render_pool.terminate()
            render_pool.join()
            renderer.destroy_window()
            return
        
    render_pool.close()
    render_pool.join()

    renderer.destroy_window()

if __name__ == '__main__':
    training_loop(parse_args())