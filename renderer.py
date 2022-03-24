import random
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

import util

import open3d as o3d
import numpy as np
from termcolor import cprint

random.seed(0)


class Renderer(ABC):
    def __init__(
            self, 
            n_primitives=50, 
            width=256, 
            height=256, 
            coordinate_scale=1.0, 
            scale_max=1.0, 
            scale_min=0.001,
            num_rotations=0,
            background_color='white',
            enable_rotations=False,
            **kwargs):
        self.n_primitives = n_primitives
        self.width = width
        self.height = height
        self.coordinate_scale = coordinate_scale
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.num_rotations = num_rotations
        self.enable_rotations = enable_rotations

        # Initialize renderer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        opt = self.vis.get_render_option()

        background = []
        if background_color == 'white':
            background = [1.0, 1.0, 1.0]
        elif background_color == 'gray' or background_color == 'grey':
            background = [0.5, 0.5, 0.5]
        elif background_color == 'black':
            background = [0.0, 0.0, 0.0]
        elif background_color == 'sky':
            background = [0.529, 0.808, 0.980]
        else:
            background = [1.0, 1.0, 1.0]
            cprint(f"Unsupported background color '{background_color}'. Defaulting to 'white'", 'yellow')

        opt.background_color = np.asarray(background)
        opt.light_on = True

        self.meshes = []
        for _ in range(self.n_primitives):          
            mesh = self.mesh_fn(create_uv_map=True)

            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()

            # Register with renderer too
            self.vis.add_geometry(mesh)
            self.meshes.append(mesh)

        # Camera initialization
        ctr = self.vis.get_view_control()
        ctr.set_lookat(np.zeros((3, 1)))
        ctr.camera_local_translate(0, 0., 0.) #front, right, top

    @property
    def n_params(self):
        return 12 # [x, y, z, r_x, r_y, r_z, s_x, s_y, s_z, r, g, b]

    @property
    @abstractmethod
    def mesh_fn():
        pass

    def _feature_rescale(self, params):
        # min-max feature scaling
        for j in range(self.n_params):
            denominator = (params[:, j].max() - params[:, j].min())
            if denominator == 0:
                continue

            if j >= 0 and j <= 2:
                # Rescale x, y, z to [-0.5, 0.5] range
                params[:, j] = -0.5 + ((params[:, j] - params[:, j].min()) * (1)) / denominator
            elif j >=3 and j <= 5:
                # Rescale r_x, r_y, r_z to [0, 360] range.
                params[:, j] = ((params[:, j] - params[:, j].min()) * (360)) / denominator
            elif j >=6 and j <= 8:
                # Use [scale_min, scale_max] range for scale features
                params[:, j] = self.scale_min + ((params[:, j] - params[:, j].min()) * (self.scale_max - self.scale_min)) / denominator
            else:
                params[:, j] = (params[:, j] - params[:, j].min()) / denominator

        return params
        

    def render(self, params, save_image="", save_rotations=False, do_absolute_scaling=True):
        params = params.copy()
        params = self._feature_rescale(params)

        for i, mesh in enumerate(self.meshes):
            # Extract individual params for this primitive
            x, y, z, r_x, r_y, r_z, s_x, s_y, s_z, r, g, b = params[i]

            x *= self.coordinate_scale
            y *= self.coordinate_scale
            z *= self.coordinate_scale


            s_x = min(1.0, max(abs(s_x), 0.01))
            s_y = min(1.0, max(abs(s_y), 0.01))
            s_z = min(1.0, max(abs(s_z), 0.01))

            mesh.paint_uniform_color([r, g, b])
            mesh.translate([x, y, z], relative=False)

            vertices = np.asarray(mesh.vertices)
            center = np.mean(vertices, axis=0)
            scales = np.asarray([s_x, s_y, s_z])

            mesh.vertices = o3d.utility.Vector3dVector(
                scales * (vertices - center) + center
            )

            if self.enable_rotations:
                R = util.get_rotation_matrix('xyz', r_x, r_y, r_z)
                mesh.rotate(R, np.array([x, y, z]))

            self.vis.update_geometry(mesh)

        self.vis.poll_events()
        self.vis.update_renderer()
        if save_image:
            self.vis.capture_screen_image(save_image)

        # Capture from multiple angles (or just the front)
        im_batch = []
        if self.num_rotations <= 1:
            im_data = np.asarray(self.vis.capture_screen_float_buffer())
            im_batch.append(im_data)
        else:
            for i in range(self.num_rotations):
                if save_rotations:
                    self.vis.capture_screen_image(f"rotation-temp-{i:03}.jpg")

                im_data = np.asarray(self.vis.capture_screen_float_buffer())
                im_batch.append(im_data)

                ctr = self.vis.get_view_control()

                # This value is a full rotation around the origin, obtained
                # after trial and error. There is no decent documentation on this
                # in Open3D.
                ctr.rotate(
                    2094.395 / self.num_rotations, 
                    0.0
                )
                self.vis.poll_events()
                self.vis.update_renderer()

        if do_absolute_scaling:
            # Rescale back up. This effectively forces the scaling to be absolute
            # w.r.t the params
            for i, mesh in enumerate(self.meshes):
                x, y, z, r_x, r_y, r_z, s_x, s_y, s_z, r, g, b = params[i]

                if self.enable_rotations:
                    R = util.get_rotation_matrix('zyx', -r_z, -r_y, -r_x)
                    mesh.rotate(R, np.array([x, y, z]))

                vertices = np.asarray(mesh.vertices)
                center = np.mean(vertices, axis=0)
                scales = np.asarray([s_x, s_y, s_z])

                mesh.vertices = o3d.utility.Vector3dVector(
                    (1 / scales) * (vertices - center) + center
                )
        
        return im_batch

    def destroy_window(self):
        self.vis.destroy_window()

    def write_meshes(self, solution, output_dir="."):
        out_file_dir = Path(output_dir)
        solution = solution.copy()
        solution = self._feature_rescale(solution)

        for i, mesh in enumerate(self.meshes):
            out_file = str(out_file_dir / f"temp_{i}.obj")
            o3d.io.write_triangle_mesh(out_file, mesh, write_ascii=True)

            # Have to update diffuse values on MTL with colors from solution.
            x, y, z, r_x, r_y, r_z, s_x, s_y, s_z, r, g, b = solution[i]
            out_mtl_file = str(out_file_dir / f"temp_{i}.mtl")
            util.update_mtl_diffuse(out_mtl_file, r, g, b)                


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

# create_uv_map not supported?
#class TorusRenderer(Renderer):
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#
#    @property
#    def mesh_fn(self):
#        return o3d.geometry.TriangleMesh.create_torus

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
                    #o3d.geometry.TriangleMesh.create_torus create_uv_map not supported?
        ]

        f = random.choice(fn)
        return f
