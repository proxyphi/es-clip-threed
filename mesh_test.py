import random
import time
import math
from typing import List

import util
import renderer

import open3d as o3d
import numpy as np

def random_mesh():
    fn = [
        o3d.geometry.TriangleMesh.create_box,
        o3d.geometry.TriangleMesh.create_cone,
        o3d.geometry.TriangleMesh.create_cylinder,
        o3d.geometry.TriangleMesh.create_icosahedron,
        o3d.geometry.TriangleMesh.create_octahedron,
        o3d.geometry.TriangleMesh.create_sphere,
        o3d.geometry.TriangleMesh.create_tetrahedron,
        o3d.geometry.TriangleMesh.create_torus
    ]

    f = random.choice(fn)
    return f()


def initialize_meshes(meshfn, n=50) -> List[o3d.geometry.TriangleMesh]:
    meshes = []
    for _ in range(n):
        mesh = meshfn()

        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.random.rand(3, 1))

        mesh.translate(np.random.rand(3, 1))
        # todo: rotation
        mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices) * np.random.rand(1, 3)
        )

        meshes.append(mesh)

    return meshes

def create_boxes(n=50) -> List[o3d.geometry.TriangleMesh]:
    return initialize_meshes(o3d.geometry.TriangleMesh.create_box, n)

def position_test() -> List[o3d.geometry.TriangleMesh]:

    mesh1 = o3d.geometry.TriangleMesh.create_box()
    mesh2 = o3d.geometry.TriangleMesh.create_box()
    mesh3 = o3d.geometry.TriangleMesh.create_box()
    mesh4 = o3d.geometry.TriangleMesh.create_box()

    meshes = [mesh1, mesh2, mesh3, mesh4]

    for mesh in meshes:
        mesh.compute_vertex_normals()

    mesh2.paint_uniform_color([1., 0., 0.])
    mesh2.translate([1., 0., 0.])

    mesh3.paint_uniform_color([0., 1., 0.])
    mesh3.translate([0., 1., 0.])

    mesh4.paint_uniform_color([0., 0., 1.])
    mesh4.translate([0., 0., 1.])

    #mesh.vertices = o3d.utility.Vector3dVector(
        #np.asarray(mesh.vertices) * np.random.rand(1, 3)
    #)

    return meshes

def scale_test() -> List[o3d.geometry.TriangleMesh]:

    mesh1 = o3d.geometry.TriangleMesh.create_box()
    mesh2 = o3d.geometry.TriangleMesh.create_box()
    mesh3 = o3d.geometry.TriangleMesh.create_box()
    mesh4 = o3d.geometry.TriangleMesh.create_box()

    meshes = [mesh1, mesh2, mesh3, mesh4]

    R1 = util.get_rotation_matrix(45, 0, 0)
    R2 = util.get_rotation_matrix(0, 45, 0)
    R3 = util.get_rotation_matrix(0, 0, 45)

    mesh2.paint_uniform_color([1., 0., 0.])
    mesh2.translate([2., 0., 0.], relative=False)
    mesh2.rotate(R1, np.array([2., 0., 0.]))
    mesh2.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh2.vertices) * [1., 1., 1.]
    )

    mesh3.paint_uniform_color([0., 1., 0.])
    mesh3.translate([0., 2., 0.,], relative=False)
    mesh3.rotate(R2, np.array([0., 2., 0.]))


    mesh3.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh3.vertices) * [1., 1., 1.]
    )

    mesh4.paint_uniform_color([0., 0., 1.])
    mesh4.translate([0., 0., 2.], relative=False)
    mesh4.rotate(R3, np.array([0., 0., 2.]))
    mesh4.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh4.vertices) * [1., 1., 1.]
    )

    for mesh in meshes:
        mesh.compute_vertex_normals()
    #mesh.vertices = o3d.utility.Vector3dVector(
        #np.asarray(mesh.vertices) * np.random.rand(1, 3)
    #)

    return meshes

def create_random_meshes(n=50) -> List[o3d.geometry.TriangleMesh]:
    return initialize_meshes(random_mesh, n)

def obj_test():
    mesh1 = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
    mesh2 = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
    mesh3 = o3d.geometry.TriangleMesh.create_sphere(create_uv_map=True)

    #test_mesh = o3d.geometry.create_box(create_uv_map=True)
    meshes = [mesh1, mesh2, mesh3]

    mesh2.paint_uniform_color([1., 0., 0.])

    mesh3.paint_uniform_color([0., 1., 0.])
    mesh3.translate([0., 2., 0.,], relative=False)
    mesh3.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh3.vertices) * [1., 2., 1.]
    )

    for i, mesh in enumerate(meshes):
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        o3d.io.write_triangle_mesh(f"test_{i}.obj", mesh, write_ascii=True)

def normals_test():
    mesh = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
    print(np.asarray(mesh.vertex_normals))
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.vertex_normals))
    mesh.normalize_normals()
    print(np.asarray(mesh.vertex_normals))

    o3d.io.write_triangle_mesh(f"normal_test.obj", mesh, write_ascii=True)

rotate_phase = 0
def rotate_view(vis):
    global rotate_phase
    ctr = vis.get_view_control()
    ctr.rotate(
        2094.395 / 4, 
        0.0
    )

    time.sleep(0.3)
    return False

#o3d.visualization.draw_geometries_with_animation_callback(scale_test(), rotate_view)
'''
meshes = scale_test()

vis = o3d.visualization.Visualizer()
vis.create_window()
for mesh in meshes:
    vis.add_geometry(mesh)

opt = vis.get_render_option()
opt.light_on = False
vis.run()
vis.destroy_window()
'''
#obj_test()
#normals_test()

render = renderer.BoxRenderer(n_primitives=1)
x = -0.5
y = -0.5
z = 0
r_x = 0
r_y = 0
r_z = 0
s_x = 0.2
s_y = 0.2
s_z = 0.2
r = 0.5
g = 0.5
b = 0.5
render.render(np.array([[x, y, z, r_x, r_y, r_z, s_x, s_y, s_z, r, g, b]]), save_image="mesh_test.png")
