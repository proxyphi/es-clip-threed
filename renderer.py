import random
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path


import open3d as o3d
import numpy as np


#class RendererOptions(ABC):
    #def __init__(self, width=256, height=256):
        #self.width = width
        #self.height = height

class Renderer(ABC):
    def __init__(self, n_primitives=50, width=256, height=256, save_image=True, coordinate_scale=1.0, **kwargs):
        self.n_primitives = n_primitives
        self.width = width
        self.height = height
        self.save_image = save_image
        self.coordinate_scale = coordinate_scale

        if save_image and not Path("./output/").exists():
            Path("./output/").mkdir()


        # Initialize renderer
        self.vis = o3d.visualization.Visualizer()
        # TODO Can visibility be toggled off but still allow screenshots?
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        self._frame_counter = 0


    @property
    @abstractmethod
    def n_params(self):
        pass 

    @abstractmethod
    def render(self, params):
        #TODO
        pass

    def destroy_window(self):
        self.vis.destroy_window()


class BoxRenderer(Renderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.meshes = []
        for _ in range(self.n_primitives):
            # Initialize params for this mesh
            x, y, z, s_x, s_y, s_z, r, g, b = np.random.rand(9,)

            mesh = o3d.geometry.TriangleMesh.create_box()

            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([r, g, b])

            mesh.translate([x, y, z])
            mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(mesh.vertices) * [s_x, s_y, s_z]
            )

            # Register with renderer too
            self.vis.add_geometry(mesh)
            self.meshes.append(mesh)

        # Camera initialization
        ctr = self.vis.get_view_control()
        ctr.set_lookat(np.zeros((3, 1)))
        ctr.camera_local_translate(-0.25, 0.5, 0.5) #front, right, top

    @property
    def n_params(self):
        return 9 # [x, y, z, s_x, s_y, s_z, r, g, b]

    def render(self, params, save_image=""):

        params = params.copy()

        # min-max feature scaling
        for j in range(self.n_params):
            params[:, j] = (params[:, j] - params[:, j].min()) / (params[:, j].max() - params[:, j].min())

        for i, mesh in enumerate(self.meshes):
            # Extract individual params for this primitive
            x, y, z, s_x, s_y, s_z, r, g, b = params[i]
            
            x *= self.coordinate_scale
            y *= self.coordinate_scale
            z *= self.coordinate_scale

            s_x = max(abs(s_x), 0.0625)
            s_y = max(abs(s_y), 0.0625)
            s_z = max(abs(s_z), 0.0625)

            mesh.paint_uniform_color([r, g, b])
            mesh.translate([x, y, z], relative=False)
            mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(mesh.vertices) * [s_x, s_y, s_z]
            )

            self.vis.update_geometry(mesh)

            # Rescale back up. This effectively forces the scaling to be absolute
            mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(mesh.vertices) / [s_x, s_y, s_z]
            )

        ctr = self.vis.get_view_control()
        ctr.rotate(
            0.0,#10.0 * random.randint(1, 24) * (-1 if random.random() > 0.5 else 1), 
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

        #self.vis.get_view_control().rotate(60., 0.) # todo callback

        return im_batch
    