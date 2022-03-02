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
from util import save_as_gif, merge_obj_files, merge_mtl_files

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
    parser.add_argument('--prompt', type=str, required=True, 
            help="The prompt that will be used to guide evolution")
    parser.add_argument('--device', type=str,
            help="Inference device. Defaults to 'cuda:0' if CUDA is supported, otherwise 'cpu'")
    parser.add_argument('--n_population', type=int, default=128,
            help="Population size of PGPE solver. Larger: Better fitness, more resource intensive.")
    parser.add_argument('--n_iterations', type=int, default=800,
            help="Number of evolutionary steps to run.")
    parser.add_argument('--n_primitives', type=int, default=50,
            help="Number of 3D geometry primitives to use.")
    parser.add_argument('--n_rotations', type=int, default=4,
            help="Number of evenly-spaced turns the camera will perform around the object "+\
                 "before completing a full turn. One screenshot is taken per angle and fed into CLIP.")
    parser.add_argument('--aug_text_input', type=bool, default=False, 
            help="If true, then the string '[aug]' when placed in the prompt will be replaced "+\
                 "with the strings 'from the front', 'from the left', 'from the back', 'from the right'. "+\
                 "This expects that n_rotations is equal to 4.")
    parser.add_argument('--recording_interval', type=int, default=20,
            help="Saves a screenshot and fitness score every recording_interval steps.")
    parser.add_argument('--batch_size', type=int, default=32,
            help="Batch size of image tensors passed into CLIP to get fitness scores. " +\
                 "Should be raised to take up available GPU memory for speed.")
    parser.add_argument('--seed', type=int, default=0, help="Seed to use for run.")
    parser.add_argument('--coordinate_scale', type=float, default=1.0, 
            help="Controls the range that primitives' XYZ coordinates can occupy.")
    parser.add_argument('--scale_max', type=float, default=0.08,
            help="Maximum scale of primitives (in relative space, absolute units)")
    parser.add_argument('--scale_min', type=float, default=0.02,
            help="Minimum scale of primitives.")
    parser.add_argument('--renderer', type=str, default='BoxRenderer',
            help="Which renderer to use. See implementing classes of Renderer in renderer.py.")
    parser.add_argument('--background_color', type=str, default='white',
            help="What color to use for background. Can be 'white' 'gray'/'grey' or 'black'")
    parser.add_argument('--enable_rotations', type=bool, default=False,
            help="Enables rotation of primitives as an optimization parameter.")
    parser.add_argument('--loss_type', type=str, default='cosine',
            help="Can be 'cosine' or 'spherical_dist_loss'.")
    parser.add_argument('--distance_weight', type=float, default=0.1,
            help="Multiplier for distance-based penalty.")

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

def fitness(batch, text_features, model, num_renders, solutions, num_augs=4, distance_weight=0.1, loss_type='cosine'):
    # Calculate penalty for center of primitives being far from the origin.
    #batch, n_prim, n_param
    # batch, n_prim -> distance from cluster center
    # batch, 3 -> cluster center
    centers = torch.mean(solutions[:, :, 0:3], dim=1)
    distance_from_origin = centers.norm(dim=-1)
    distance_penalty = distance_weight * distance_from_origin
    distance_penalty = distance_penalty.repeat_interleave(num_augs, dim=0)

    with torch.no_grad():
        image_features = model.encode_image(batch)
        # Need to tile text augs to match up with camera angles
        if text_features.shape[0] > 1:
            n_repeats = image_features.shape[0] // text_features.shape[0]
            text_features = torch.tile(text_features, (n_repeats, 1))
        if loss_type == 'cosine':
            fit = torch.cosine_similarity(image_features, text_features, axis=-1)
        elif loss_type == 'spherical_dist_loss':
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Make negative since we're maximizing.
            fit = -(image_features - text_features).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

        fit = fit - distance_penalty
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
        'num_rotations': args.n_rotations,
        'background_color': args.background_color,
        'enable_rotations': args.enable_rotations
    }
    return renderer_args

def get_renderer_class(name):
    renderer = __import__('renderer')
    return getattr(renderer, name)

### For multiprocessing pool. ###
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

#################################

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

def write_obj_out(args, solution, out_file):
    renderer_cls = get_renderer_class(args.renderer)
    renderer_args = get_renderer_args(args)
    temp_renderer = renderer_cls(**renderer_args)

    temp_renderer.render(solution, do_absolute_scaling=False)
    temp_renderer.write_meshes(solution)

    merge_obj_files("temp_*.obj", out_file)
    out_mat_file = out_file[:-4] + '.mtl'
    merge_mtl_files("temp_*.mtl", out_mat_file)

    # Delete all temp obj files.
    for f in glob.glob("*.obj") + glob.glob("*.mtl"):
        try:
            Path(f).unlink()
        except FileNotFoundError:
            continue


def main(args):
    if args.aug_text_input and args.n_rotations != 4:
        cprint("aug_text_input requires that n_rotations == 4.", "red")
        return

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
        optimizer_config= dict(
            max_speed=0.2,
            momentum=0.9
        ),
        seed=args.seed
    )

    # Initialize CLIP model + Prompt embedding
    clip_model = clip.load('ViT-B/16', jit=True, device=device)[0]

    text_inputs = []

    if args.aug_text_input:
        text_inputs.append(clip.tokenize(args.prompt.replace('[aug]', 'from the front')).to(device))
        text_inputs.append(clip.tokenize(args.prompt.replace('[aug]', 'from the left side')).to(device))
        text_inputs.append(clip.tokenize(args.prompt.replace('[aug]', 'from the back')).to(device))
        text_inputs.append(clip.tokenize(args.prompt.replace('[aug]', 'from the right side')).to(device))
    else:
        text_inputs.append(clip.tokenize(args.prompt).to(device))

    text_features = []
    with torch.no_grad():
        for text_input in text_inputs:
            text_features.append(clip_model.encode_text(text_input))
        text_features = torch.concat(text_features).to(device) # need to look this up

    render_pool = mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(args,))

    # Evolutionary loop
    n_iterations = args.n_iterations
    n_captures = args.n_rotations
    recording_interval = args.recording_interval
    for i in trange(n_iterations):
        try:
            solutions = solver.ask()
            solutions = [s.reshape((renderer.n_primitives, renderer.n_params)) for s in solutions]
            
            # Render each solution from the solver
            renders = render_pool.map(func=render_fn, iterable=solutions)
            renders = [render for render in renders]

            # Process and augment all renders
            im_batch = [process_augment_renders(r, device) for r in renders]

            # Rearrange im_batch into even batch_size chunks
            n_chunks = math.ceil((args.n_population * n_captures) / args.batch_size)

            im_batch = torch.stack(im_batch, dim=0).reshape(-1, 3, 224, 224)
            im_batches = torch.chunk(im_batch, n_chunks) # n_chunks x [batch, 3, 224, 224]

            chunk_size = args.n_population // n_chunks    

            solutions = torch.tensor(np.array(solutions)).to(device)

            fitnesses = [
                fitness(batch, text_features, clip_model, num_renders=chunk_size, 
                        solutions=solutions, num_augs=n_captures, loss_type=args.loss_type) 
                for batch in im_batches
            ]
            fitnesses = np.concatenate(fitnesses)

            # Run evolutionary step
            solver.tell(fitnesses)

            # Recording interval hooks
            if (i+1) % recording_interval == 0 or i == 0:
                best_solution = solver.center.reshape(renderer.n_primitives, renderer.n_params)
                save_image_name = str(output_dir / f"frame_{i+1:04}.jpg")
                renderer.render(best_solution, save_image=save_image_name)
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

    # Write out OBJ of best solution.
    write_obj_out(
        args,
        best_solution,
        str(output_dir / "output.obj")
    )

if __name__ == '__main__':
    main(parse_args())