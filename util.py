import glob
import copy
from pathlib import Path
from typing import NamedTuple

from scipy.spatial.transform import Rotation
from PIL import Image

class OBJMetadata(NamedTuple):
    object_name: str
    n_vertices: int
    n_triangles: int
    mtl_file: str

def save_as_gif(fn, fp_in, fps=24):
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    with open(fn, 'wb') as fp_out:
        img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=int(1000./fps), loop=0)

def get_vertex_section_start(obj_filedata):
    if obj_filedata[4].split(' ')[0] == 'mtllib':
        return 5

    return 4

def get_face_section_start(obj_filedata):
    for i, line in enumerate(obj_filedata):
        if line[0:6] == 'usemtl':
            return i
        elif line[0] == 'f':
            return i

def get_metadata(obj_filedata) -> OBJMetadata:
    object_name = obj_filedata[1].split('object name: ')[-1]
    n_vertices = int(obj_filedata[2].split(' ')[-1])
    n_triangles = int(obj_filedata[3].split(' ')[-1])

    mtl_file = None
    if obj_filedata[4].split(' ')[0] == 'mtllib':
        mtl_file = obj_filedata[4].split(' ')[-1]

    return OBJMetadata(object_name, n_vertices, n_triangles, mtl_file)

def merge_metadata_sections(obj_filedata1, obj_filedata2):
    metadata1 = get_metadata(obj_filedata1)
    metadata2 = get_metadata(obj_filedata2)

    n_vertices = metadata1.n_vertices + metadata2.n_vertices
    n_triangles = metadata1.n_triangles + metadata2.n_triangles

    obj_filedata1[2] = obj_filedata1[2].replace(str(metadata1.n_vertices), str(n_vertices))
    obj_filedata1[3] = obj_filedata1[3].replace(str(metadata1.n_triangles), str(n_triangles))

def merge_vertex_sections(obj_filedata1, obj_filedata2):
    vertex_end1 = get_face_section_start(obj_filedata1)

    vertex_start2 = get_vertex_section_start(obj_filedata2)
    vertex_end2 = get_face_section_start(obj_filedata2)

    # Extends obj_filedata1 at point vertex_end1
    obj_filedata1[vertex_end1:vertex_end1] = obj_filedata2[vertex_start2:vertex_end2]

def merge_face_sections(obj_filedata1, obj_filedata2):
    n_vertices1 = get_metadata(obj_filedata1).n_vertices
    face_start2 = get_face_section_start(obj_filedata2)

    if obj_filedata2[face_start2].split(' ')[0] == 'usemtl':
        obj_filedata1.append(obj_filedata2[face_start2])
        face_start2 += 1

    updated_faces = []
    for face_data in obj_filedata2[face_start2:]:
        new_face = copy.copy(face_data)
        new_face = new_face.rstrip()
        new_face = new_face.split(' ')

        def update_face_vert(face_vert, n_vertices):
            if face_vert == 'f':
                return 'f'
            
            first_vert, second_vert, third_vert = face_vert.split('/')
            return '/'.join([
                str(int(first_vert) + n_vertices),
                '' if second_vert == '' else str(int(second_vert) + n_vertices),
                str(int(third_vert) + n_vertices)
            ])
            
        # Adds n_vertices to each vertex index in the face.
        new_face = list(map(update_face_vert, new_face, [n_vertices1] * 4))
        new_face = ' '.join(new_face)
        new_face += '\n'

        updated_faces.append(new_face)

    obj_filedata1.extend(updated_faces)

def merge_obj_files(fp_in, out_file):
    out_file_data = []
    for obj_filename in glob.glob(fp_in):
        with Path(obj_filename).open('r') as obj_file:
            obj_data = obj_file.readlines()

            if len(out_file_data) == 0:
                out_file_data = copy.copy(obj_data)
                continue

            merge_vertex_sections(out_file_data, obj_data)
            merge_face_sections(out_file_data, obj_data)
            merge_metadata_sections(out_file_data, obj_data)
    
    with Path(out_file).open('w+') as f:
        f.writelines(out_file_data)

def merge_mtl_files(fp_in, out_file):
    out_file_data = []
    for mtl_filename in glob.glob(fp_in):
        with Path(mtl_filename).open('r') as mtl_file:
            mtl_data = mtl_file.readlines()

            if len(out_file_data) == 0:
                out_file_data = copy.copy(mtl_data)
                out_file_data += '\n'
                continue

            out_file_data.extend(mtl_data[2:])
            out_file_data += '\n'
    
    with Path(out_file).open('w+') as f:
        f.writelines(out_file_data)

def update_mtl_diffuse(mtl_filename, r, g, b):
    with Path(mtl_filename).open('r+') as f:
        mtl_data = f.readlines()
        diffuse_line_idx = -1
        diffuse_line = f"Kd {r:.4f} {g:.4f} {b:.4f}\n"

        for i, line in enumerate(mtl_data):
            if line[:2] == 'Kd':
                # Found diffuse line.
                diffuse_line_idx = i
                break

        mtl_data[diffuse_line_idx] = diffuse_line
        f.truncate()
        f.seek(0)
        f.writelines(mtl_data)

def get_rotation_matrix(r_x, r_y, r_z):
    R = Rotation.from_euler('zyx', [r_z, r_y, r_x], degrees=True)
    return R.as_matrix()