# CLIP-Guided 3D Scene Evolution


![](/assets/output-rotating-rubberduck.gif)
![](/assets/output-rubberduck.gif)
![](/assets/output-rotating.gif)
![](/assets/output-darwin.gif)

This project aims to generate 3D scenes using prompts fed into OpenAI's CLIP to guide how various primitives should be arranged within a scene. The overall idea can just be considered an extension of [Modern Evolution Strategies for Creativity: Fitting Concrete Images and Abstract Concepts](https://es-clip.github.io/) to three dimensions.

The following features are supported:

* Changing what types of primitives are used (e.g only boxes, spheres, or use random primitives.). See `renderer.py` for supported primitives.
* Whether or not to allow primitives to rotate in addition to their XYZ position, XYZ scales, and RGB color being optimizeable
* Modifying the range of valid scales or valid XYZ positions of all primitives
* Exporting the final result as a `.obj` file with corresponding `.mtl` file, which can be viewed in a supporting 3D viewer.

Also as part of this, a full writeup documenting my experience working on this will be published and linked at some point. This was as kind of a personal experiment to document and refine my productive process, but figured someone else might find it interesting.

TODO: Document more results and examples!

## Requirements
* Python 3.6+
* [PyTorch](https://pytorch.org) with CUDA support (1.7.1+ tested)
* [PGPElib](https://github.com/nnaisense/pgpelib).
* [CLIP](https://github.com/openai/CLIP) from OpenAI.

Runs on Windows 10 / 11. Untested with Linux / macOS.
## Installation
Install PyTorch above first, and then the remaining requirements with:
```
pip install -r requirements.txt
```
PGPElib and CLIP should be installed as well, but if not, you'll need to install
those too.

## Usage
You can specify a prompt to use for generation as follows:
```bash
python main.py --prompt "A picture of a rubber duck"
```
This will create an output directory automatically with your results in it. By default, screenshots of progress are taken every 20 steps of evolution, and
at the end of the evolutionary loop a .obj & .mtl file are written out. I find these are best viewed in something like Blender.

If you want to instead fit a target image, you can use:
```bash
python l2_loss.py --target "[filepath to target image]"
```
The image will automatically be resized to 256x256.

Use `python main.py -h` or `python l2_loss.py -h` to see a full list of supported arguments.

## References
1. For the overall idea and inspiration: https://es-clip.github.io/ [https://arxiv.org/pdf/2109.08857.pdf]

2. For PGPE and its documentation: https://github.com/nnaisense/pgpelib

3. For inspiration and definition of `spherical_dist_loss` https://github.com/crowsonkb/v-diffusion-pytorch

4. [Open3D](http://www.open3d.org/) for doing most of the heavy lifting for rendering 3D scenes and generating primitives.
5. http://paulbourke.net/dataformats/ for great explanations on the [OBJ](http://paulbourke.net/dataformats/obj) and [MTL](http://paulbourke.net/dataformats/mtl) file formats.
