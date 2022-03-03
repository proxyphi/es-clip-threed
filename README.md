# CLIP Guided Evolution for Generating 3D Scenes by Arranging Simple Primitives

TODO: Write me!

## Requirements
* Python 3.6+
* [pytorch](https://pytorch.org) with CUDA support
* [PGPElib](https://github.com/nnaisense/pgpelib).
* [CLIP](https://github.com/openai/CLIP) from OpenAI.

## Installation
Install the PyTorch above first, and then the remaining requirements with:
```
pip install -r requirements.txt
```
PGPElib and CLIP should be installed as well, but if not, you'll need to install
those as well.

## Usage
You can specify a prompt to use for generation as follows:
```bash
python main.py --prompt "A picture of a rubber duck"
```
This will create an output directory automatically with your results in it. By default, screenshots of progress are taken every 20 steps of evolution, and
at the end of the evolutionary loop a .obj & .mtl file are written out. I find these are best viewed in something like Blender.