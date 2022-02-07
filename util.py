import glob
import time
from PIL import Image

def save_as_gif(fn, fp_in, fps=24):
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    with open(fn, 'wb') as fp_out:
        img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=int(1000./fps), loop=0)

if __name__ == '__main__':
    save_as_gif("./output/output.gif", "./output/*.jpg")
