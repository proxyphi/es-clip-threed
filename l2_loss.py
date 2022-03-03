import argparse
import multiprocessing as mp
import string
import glob
import datetime
from typing import List
from pathlib import Path

from renderer import *

import numpy as np
from tqdm import trange
from pgpelib import PGPE
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True,
            help="File path to an image to fit.")
    parser.add_argument('--device', type=str,
            help="Inference device. Defaults to 'cuda:0' if CUDA is supported, otherwise 'cpu'")
    parser.add_argument('--n_population', type=int, default=32,
            help="Population size of PGPE solver. Larger: Better fitness, more resource intensive.")
    parser.add_argument('--n_iterations', type=int, default=800,
            help="Number of evolutionary steps to run.")
    parser.add_argument('--n_primitives', type=int, default=50,
            help="Number of 3D geometry primitives to use.")
    parser.add_argument('--recording_interval', type=int, default=20,
            help="Saves a screenshot and fitness score every recording_interval steps.")
    parser.add_argument('--seed', type=int, default=0, help="Seed to use for run.")
    parser.add_argument('--coordinate_scale', type=float, default=1.0, 
            help="Controls the range that primitives' XYZ coordinates can occupy.")
    parser.add_argument('--scale_max', type=float, default=0.1,
            help="Maximum scale of primitives (in relative space, absolute units)")
    parser.add_argument('--scale_min', type=float, default=0.02,
            help="Minimum scale of primitives.")
    parser.add_argument('--renderer', type=str, default='BoxRenderer',
            help="Which renderer to use. See implementing classes of Renderer in renderer.py.")
    parser.add_argument('--background_color', type=str, default='white',
            help="What color to use for background. Can be 'white' 'gray'/'grey', 'black', or 'sky'")
    parser.add_argument('--enable_rotations', type=bool, default=False,
            help="Enables rotation of primitives as an optimization parameter.")

    args = parser.parse_args()
    return args

def setup_output_dir(args) -> Path:
    output_dir = Path("./output/")
    if not output_dir.exists():
        output_dir.mkdir()
    
    target = args.target
    
    # Removes all punctuation
    dir_name = target.translate(target.maketrans("", "", string.punctuation))

    # Extract the first three words in the target and replace with _
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
    renderer_cls = util.get_renderer_class(args.renderer)
    renderer_args = util.get_renderer_args(args)
    renderer_args['num_rotations'] = 30

    temp_renderer = renderer_cls(**renderer_args)

    temp_renderer.render(solution, save_rotations=True)
    util.save_as_gif(out_file, "rotation-temp-*.jpg")

    for f in glob.glob("rotation-temp-*.jpg"):
        try:
            Path(f).unlink()
        except FileNotFoundError:
            continue

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


# For multiprocessing pool.
worker_renderer = None
def init_worker(args):
    global worker_renderer
    renderer_cls = util.get_renderer_class(args.renderer)
    renderer_args = util.get_renderer_args(args)
    worker_renderer = renderer_cls(**renderer_args)

def render_fn(params):
    global worker_renderer
    render = worker_renderer.render(params)
    return render

def main(args):
    args.n_rotations = 1
    np.random.seed(args.seed)

    output_dir = setup_output_dir(args)

    # Initialize Renderer
    renderer_cls = util.get_renderer_class(args.renderer)
    renderer_args = util.get_renderer_args(args)
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
            renders = render_pool.map(func=render_fn, iterable=solutions)
            renders = [render for render in renders]

            # Process all renders
            im_batch = [process_renders(r)[0, :] for r in renders]

            # Calculate fitness
            fitnesses = [fitness(batch, target_image, loss_fn=l2_loss) for batch in im_batch]

            solver.tell(fitnesses)

            # Recording interval hooks
            if (i+1) % args.recording_interval == 0 or i == 0:
                best_solution = solver.center.reshape(renderer.n_primitives, renderer.n_params)
                save_image_name = str(output_dir / f"frame_{i+1:04}.jpg")
                renderer.render(best_solution, save_image=save_image_name)
                with (output_dir / "fitnesses.txt").open('a') as f:
                    f.write(f"[{datetime.datetime.now()}] Iteration: {i+1} Fitness: {np.max(fitnesses):.8f}\n")
            
        except KeyboardInterrupt:
            break

    render_pool.close()
    render_pool.join()

    renderer.destroy_window()

    # To view the final evolved model rotating.
    do_final_render(
        args,
        best_solution,
        str(output_dir /"output-rotating.gif"))

    # To save evolution progress as gif
    util.save_as_gif(str(output_dir / "output.gif"), str(output_dir / "*.jpg"))

if __name__ == '__main__':
    main(parse_args())