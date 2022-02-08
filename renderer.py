import random
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path


import open3d as o3d
import numpy as np

random.seed(0)

#class RendererOptions(ABC):
    #def __init__(self, width=256, height=256):
        #self.width = width
        #self.height = height

class Renderer(ABC):
    def __init__(
            self, 
            n_primitives=50, 
            width=256, 
            height=256, 
            coordinate_scale=1.0, 
            scale_max=1.0, 
            scale_min=0.001,
            random_rotate=False,
            **kwargs):
        self.n_primitives = n_primitives
        self.width = width
        self.height = height
        self.coordinate_scale = coordinate_scale
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.random_rotate = random_rotate

        # Initialize renderer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        self._frame_counter = 0

        self.meshes = []
        for _ in range(self.n_primitives):          
            mesh = self.mesh_fn()

            mesh.compute_vertex_normals()

            # Register with renderer too
            self.vis.add_geometry(mesh)
            self.meshes.append(mesh)

        # Camera initialization
        ctr = self.vis.get_view_control()
        ctr.set_lookat(np.zeros((3, 1)))
        ctr.camera_local_translate(0, 0., 0.) #front, right, top

    @property
    def n_params(self):
        return 9 # [x, y, z, s_x, s_y, s_z, r, g, b]

    @property
    @abstractmethod
    def mesh_fn():
        pass

    def render(self, params, save_image=""):
        params = params.copy()

        # min-max feature scaling
        for j in range(self.n_params):
            if j >= 0 and j <= 2:
                # Rescale x, y, z to [-1, 1] range
                params[:, j] = -1 + ((params[:, j] - params[:, j].min()) * (2)) / (params[:, j].max() - params[:, j].min())
            elif j >=3 and j <= 5:
                # Use [scale_min, scale_max] range for scale features
                params[:, j] = self.scale_min + ((params[:, j] - params[:, j].min()) * (self.scale_max - self.scale_min)) / (params[:, j].max() - params[:, j].min())
            else:
                params[:, j] = (params[:, j] - params[:, j].min()) / (params[:, j].max() - params[:, j].min())

        for i, mesh in enumerate(self.meshes):
            # Extract individual params for this primitive
            x, y, z, s_x, s_y, s_z, r, g, b = params[i]

            x *= self.coordinate_scale
            y *= self.coordinate_scale
            z *= self.coordinate_scale

            s_x = min(1.0, max(abs(s_x), 0.01))
            s_y = min(1.0, max(abs(s_y), 0.01))
            s_z = min(1.0, max(abs(s_z), 0.01))

            mesh.paint_uniform_color([r, g, b])
            mesh.translate([x, y, z], relative=False)
            mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(mesh.vertices) * [s_x, s_y, s_z]
            )

            self.vis.update_geometry(mesh)

        if self.random_rotate:
            ctr = self.vis.get_view_control()
            ctr.rotate(
                10.0 * random.randint(1, 24) * (-1 if random.random() > 0.5 else 1), 
                0.0
            )

        self.vis.poll_events()
        self.vis.update_renderer()
        if save_image:
            self.vis.capture_screen_image(save_image)
            self._frame_counter += 1

        im_batch = []
        for _ in range(1):
            im_data = np.asarray(self.vis.capture_screen_float_buffer())
            im_batch.append(im_data)

        for i, mesh in enumerate(self.meshes):
            # Extract individual params for this primitive
            x, y, z, s_x, s_y, s_z, r, g, b = params[i]
            
            s_x = min(1.0, max(abs(s_x), 0.01))
            s_y = min(1.0, max(abs(s_y), 0.01))
            s_z = min(1.0, max(abs(s_z), 0.01))

            # Rescale back up. This effectively forces the scaling to be absolute
            mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(mesh.vertices) / [s_x, s_y, s_z]
            )
        
        return im_batch

    def destroy_window(self):
        self.vis.destroy_window()


class BoxRenderer(Renderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def mesh_fn(self):
        return o3d.geometry.TriangleMesh.create_box

class SphereRenderer(Renderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def mesh_fn(self):
        return o3d.geometry.TriangleMesh.create_sphere

class TorusRenderer(Renderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def mesh_fn(self):
        return o3d.geometry.TriangleMesh.create_torus

class RandomRenderer(Renderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def mesh_fn(self):
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
        return f
