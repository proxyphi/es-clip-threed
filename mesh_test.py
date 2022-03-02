import random
import time
import math
from typing import List

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

    

    mesh2.paint_uniform_color([1., 0., 0.])
    mesh2.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh2.vertices) * [-2., 1., 1.]
    )

    mesh3.paint_uniform_color([0., 1., 0.])
    mesh3.translate([0., 2., 0.,], relative=False)
    print(np.asarray(mesh3.vertices))
    print(np.asarray(mesh3.vertices).shape)
    mesh3.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh3.vertices) * [1., 2., 1.]
    )

    mesh4.paint_uniform_color([0., 0., 1.])
    mesh4.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh4.vertices) * [1., 1., 2.]
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
    #test_mesh.triangles
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

o3d.visualization.draw_geometries_with_animation_callback(scale_test(), rotate_view)
#obj_test()
#normals_test()