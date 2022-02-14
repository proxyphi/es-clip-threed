import argparse
import math
import glob
import time
import string
import datetime
import multiprocessing as mp
from typing import List
from pathlib import Path

from renderer import *
from util import save_as_gif

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
    parser.add_argument('--n_rotations', type=int, default=8)
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

def process_augment_renders(renders: List[np.ndarray], device: str):
    t = np.stack(renders, axis=0).transpose(0, 3, 1, 2)
    t = torch.tensor(t).to(device)
    t = t.type(torch.float32)
    #t = t.repeat_interleave(num_augs, dim=0)
    new_augment_trans = transforms.Compose([
        transforms.Resize(224),
        #transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
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
        'num_rotations': args.n_rotations
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


def setup_output_dir(args) -> Path:
    output_dir = Path("./output/")
    if not output_dir.exists():
        output_dir.mkdir()
    
    prompt = args.prompt
    
    # Removes all punctuation
    dir_name = prompt.translate(prompt.maketrans("", "", string.punctuation))

    # Extract the first three words in the prompt and replace with _
    dir_name = '_'.join(dir_name.split(' ')[:3])

    # Append today's date
    today = datetime.date.today()
    dir_name = f'{dir_name}_{today.year}{today.month:02}{today.day:02}'

    # Get a number for this run
    i = 0
    dir_path = output_dir / Path(dir_name + f'_{i:04}')
    while dir_path.exists():
        i += 1
        dir_path = output_dir / Path(dir_name + f'_{i:04}')

    dir_path.mkdir()

    with (dir_path / "run_args.txt").open('w+') as f:
        for arg_name, value in vars(args).items():
            f.write(f"{arg_name}: {value}\n")

    with (dir_path / "fitnesses.txt").open('w+') as f:
        f.truncate(0)
    
    return dir_path

def do_final_render(args, solution, out_file):
    renderer_cls = get_renderer_class(args.renderer)
    renderer_args = get_renderer_args(args)
    renderer_args['num_rotations'] = 30

    temp_renderer = renderer_cls(**renderer_args)

    temp_renderer.render(solution, save_rotations=True)
    save_as_gif(out_file, "rotation-temp-*.jpg")

    for f in glob.glob("rotation-temp-*.jpg"):
        try:
            Path(f).unlink()
        except FileNotFoundError:
            continue

def main(args):
    output_dir = setup_output_dir(args)

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

    render_pool = mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(args,))

    # Evolutionary loop
    n_iterations = args.n_iterations
    n_captures = args.n_rotations
    recording_interval = 20
    for i in trange(n_iterations):
        try:
            solutions = solver.ask()
            solutions = [s.reshape((renderer.n_primitives, renderer.n_params)) for s in solutions]
            
            # Render each solution from the solver
            t1 = time.time()
            renders = render_pool.map(func=render_fn, iterable=solutions)
            renders = [render for render in renders]
            t2 = time.time()
            #print(f"Rendering time: {(t2 - t1):.4f}s")

            # Process and augment all renders
            t1 = time.time()
            im_batch = [process_augment_renders(r, device) for r in renders]
            t2 = time.time()
            #print(f"Processing / Augmenting time: {(t2 - t1):.4f}s")

            # Rearrange im_batch into even batch_size chunks
            n_chunks = math.ceil((args.n_population * n_captures) / args.batch_size)

            im_batch = torch.stack(im_batch, dim=0).reshape(-1, 3, 224, 224)
            im_batches = torch.chunk(im_batch, n_chunks) # n_chunks x [batch, 3, 224, 224]

            chunk_size = args.n_population // n_chunks    

            t1 = time.time()    
            fitnesses = [fitness(batch, text_features, clip_model, num_renders=chunk_size, num_augs=n_captures, loss_type=args.loss_type) for batch in im_batches]
            t2 = time.time()
            #print(f"CLIP Fitness calculation time: {(t2 - t1):.4f}s")
            fitnesses = np.concatenate(fitnesses)

            t1 = time.time()
            solver.tell(fitnesses)
            t2 = time.time()
            #print(f"Solver time: {(t2 - t1):.4f}s")

            # TODO This can be replaced by hooks.
            if (i+1) % recording_interval == 0 or i == 0:
                best_solution = solver.center.reshape(renderer.n_primitives, renderer.n_params)
                save_image_name = str(output_dir / f"frame_{i+1:04}.jpg")
                renderer.render(best_solution, save_image=save_image_name)
                cprint(f"Fitness: {np.max(fitnesses):.8f}", "green")
                with (output_dir / "fitnesses.txt").open('a') as f:
                    f.write(f"[{datetime.datetime.now()}] Iteration: {i+1} Fitness: {np.max(fitnesses):.8f}\n")
                
        except KeyboardInterrupt:
            render_pool.terminate()
            render_pool.join()
            renderer.destroy_window()
            return
        
    render_pool.close()
    render_pool.join()

    renderer.destroy_window()

    # To view the final evolved model rotating.
    do_final_render(
        args,
        best_solution,
        str(output_dir /"output-rotating.gif"))

    # To save evolution progress as gif
    save_as_gif(str(output_dir / "output.gif"), str(output_dir / "*.jpg"))

if __name__ == '__main__':
    main(parse_args())