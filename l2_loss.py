import argparse
import multiprocessing as mp
import time
from datetime import datetime
from typing import List
from pathlib import Path

from renderer import *

import torch

import numpy as np
from tqdm import trange
from pgpelib import PGPE
from termcolor import cprint
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--device', type=str)
    parser.add_argument('--n_population', type=int, default=128)
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--n_primitives', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--coordinate_scale', type=float, default=1.0)
    parser.add_argument('--scale_max', type=float, default=1.0)
    parser.add_argument('--scale_min', type=float, default=0.01)
    parser.add_argument('--renderer', type=str, default='BoxRenderer')

    args = parser.parse_args()
    return args

def rgba2rgb(rgba_img):
    h, w = rgba_img.size
    rgb_img = Image.new('RGB', (h, w))
    rgb_img.paste(rgba_img)
    return rgb_img

def img2arr(img):
    return np.array(img)

def load_target(fn, resize):
    img = Image.open(fn)
    img = rgba2rgb(img)
    h, w = resize
    img = img.resize((w, h), Image.LANCZOS)
    img_arr = img2arr(img)
    img_arr = img_arr.astype(np.float32) / 255.0

    return img_arr

def process_renders(renders: List[np.ndarray]):
    t = np.stack(renders, axis=0)
    return t

def l2_loss(x_arr, target_arr):
    loss = (target_arr - x_arr)**2
    loss = loss.mean()
    return loss

def fitness(x_arr, target_arr, loss_fn=l2_loss):
    return -loss_fn(x_arr, target_arr) # Negative for PGPE to maximize

def get_renderer_args(args):
    renderer_args = {
        'n_primitives': args.n_primitives,
        'width': 256,
        'height': 256,
        'coordinate_scale': args.coordinate_scale,
        'scale_max': args.scale_max,
        'scale_min': args.scale_min
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
        optimizer_config={'max_speed': 0.15},
        seed=args.seed
    )

    # Load target image
    target_image = load_target(args.target, (256, 256))

    with Path("output/fitnesses.txt").open('w+') as f:
        f.truncate(0)

    render_pool = mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(args,))

    # Evolutionary loop
    n_iterations = args.n_iterations
    
    for i in trange(n_iterations):
        try:
            solutions = solver.ask()
            solutions = [s.reshape((renderer.n_primitives, renderer.n_params)) for s in solutions]
            
            # Render each solution from the solver
            t1 = time.time()
            renders = render_pool.map(func=render_fn, iterable=solutions)
            renders = [render for render in renders]
            #renders = [renderer.render(s) for s in solutions]
            t2 = time.time()
            #print(f"Rendering time: {(t2 - t1):.4f}s")

            # Process all renders
            t1 = time.time()
            im_batch = [process_renders(r)[0, :] for r in renders]
            t2 = time.time()
            #print(f"Processing / Augmenting time: {(t2 - t1):.4f}s")

            # Calculate fitness
            t1 = time.time()    
            fitnesses = [fitness(batch, target_image, loss_fn=l2_loss) for batch in im_batch]
            t2 = time.time()
            #print(f"L2 Fitness calculation time: {(t2 - t1):.4f}s")

            t1 = time.time()
            solver.tell(fitnesses)
            t2 = time.time()
            #print(f"Solver time: {(t2 - t1):.4f}s")

            # TODO This can be replaced by hooks.
            if i % 20 == 0:
                best_solution = solver.center.reshape(renderer.n_primitives, renderer.n_params)
                renderer.render(best_solution, save_image=f"./output/frame_{i:04}.jpg")
                cprint(f"Fitness: {np.max(fitnesses):.8f}", "green")
                with Path("output/fitnesses.txt").open('a') as f:
                    f.write(f"[{datetime.now()}] Iteration: {i} Fitness: {np.max(fitnesses):.8f}\n")
        except KeyboardInterrupt:
            break

    #print(solver.center)
    render_pool.close()
    render_pool.join()

    renderer.destroy_window()
    pass

if __name__ == '__main__':
    training_loop(parse_args())