from sapien.utils import Viewer
import sapien.core as sapien
import numpy as np
import importlib
from gym import Env, spaces
from sapien.asset import download_partnet_mobility
from sapien.core import Pose
from sapien_rl.utils.config_parser import (
    preprocess,
    process_variables,
    process_variants,
)
from sapien_rl.utils.misc import log_info
import warnings
from copy import deepcopy
import os
#from ..agent import CombinedAgent
import transforms3d
from transforms3d.quaternions import quat2mat, mat2quat
from collections import defaultdict


_engine = sapien.Engine()
_renderer = sapien.VulkanRenderer(default_mipmap_levels=4)
_engine.set_renderer(_renderer)


class SapienEnv(Env):
    def __init__(self, config_file, obs_mode="state", observer='world', frame_skip=1, variant_config={}, skip_reward=False):
        self.skip_reward = skip_reward

        self._engine = _engine
        self._renderer = _renderer

        self._setup_main_rng(seed=np.random.seed())

        self.obs_mode = obs_mode # state/rgbd/pointcloud
        self.observer = observer # world/robot/...
        self.variant_config = variant_config
        self.frame_skip = frame_skip

        self.yaml_config = preprocess(config_file)

        self.simulation_frequency = self.yaml_config["physics"]["simulation_frequency"]
        self.timestep = 1 / self.simulation_frequency

        self.configure_env()
        obs = self.reset(level=0)

        self.action_space = spaces.Box(low=-1, high=1, shape=self.agent.action_range().shape)
        self.observation_space = self._observation_to_space(obs)
        self._viewer = None

    def configure_env(self):
        pass

    def reset_level(self):
        self.agent.reset()
        for articulation in self.articulations.values():
            articulation["articulation"].unpack(articulation["init_state"])
        for actor in self.actors.values():
            actor["actor"].unpack(actor["init_state"])
        return self.get_obs()

    def reset(self, level=None):
        if level is None:
            level = self._main_rng.randint(2 ** 32)
        self.level = level
        self._level_rng = np.random.RandomState(seed=self.level)

        # recreate scene
        scene_config = sapien.SceneConfig()
        for p, v in self.yaml_config["physics"].items():
            if p != "simulation_frequency":
                setattr(scene_config, p, v)
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_timestep(self.timestep)

        config = deepcopy(self.yaml_config)
        config = process_variables(config, self._level_rng)
        self.level_config, self.level_variant_config = process_variants(
            config, self._level_rng, self.variant_config
        )

        # load everything
        self._setup_renderer()
        self._setup_physical_materials()
        self._setup_render_materials()
        self._load_actors()
        self._load_articulations()
        self._setup_objects()
        self._load_agent()
        self._load_custom()
        self._setup_cameras()

        self._init_eval_record()


        # Cannot return obs right now because something will be determined in derived class
        # return self.get_obs() 

    def _init_eval_record(self):
        self.keep_good_steps = defaultdict(int)
        keep_good_time = 1.0
        self.keep_good_steps_threshold = int( np.ceil(keep_good_time * self.agent.control_frequency / self.frame_skip) ) 

    def _setup_main_rng(self, seed):
        self._main_seed = seed
        self._main_rng = np.random.RandomState(seed)

    def _setup_renderer(self):
        self._scene.set_ambient_light(
            self.level_config["render"]["ambient_light"]["color"]
        )
        for pl in self.level_config["render"]["point_lights"]:
            self._scene.add_point_light(pl["position"], pl["color"])
        for dl in self.level_config["render"]["directional_lights"]:
            self._scene.add_directional_light(dl["direction"], dl["color"])

    def _setup_cameras(self):
        cameras = []
        for cam_info in self.level_config["render"]["cameras"]:
            cam_info = deepcopy(cam_info)
            if "parent" in cam_info:
                if cam_info["parent"] == 'robot':
                    parent = self.agent.get_base_link()
                else:
                    parent = self.objects[cam_info["parent"]]
                    if isinstance(parent, sapien.Articulation):
                        parent = parent.get_links()[0]
                camera_mount_actor = parent
                del cam_info["parent"]
            else:
                camera_mount_actor = self._scene.create_actor_builder().build_kinematic()
            pose = sapien.Pose(cam_info["position"], cam_info["rotation"])
            del cam_info["position"], cam_info["rotation"]
            # camera = self._scene.add_mounted_camera(
            #     actor=camera_mount_actor, pose=sapien.Pose(), **cam_info, fovx=0
            # )
            # camera_mount_actor.set_pose(pose)
            camera = self._scene.add_mounted_camera(
                actor=camera_mount_actor, pose=pose, **cam_info, fovx=0
            )
            cameras.append(camera)
        self.cameras = cameras

    def _setup_physical_materials(self):
        self.physical_materials = {}
        if "surface_materials" in self.level_config["layout"]:
            for material in self.level_config["layout"]["surface_materials"]:
                m = self._scene.create_physical_material(
                    material["static_friction"],
                    material["dynamic_friction"],
                    material["restitution"],
                )
                self.physical_materials[material["name"]] = m

    def _setup_render_materials(self):
        self.render_materials = {}
        if "materials" in self.level_config["render"]:
            for material in self.level_config["render"]["materials"]:
                m = self._renderer.create_material()
                m.set_roughness(material["roughness"])
                m.set_specular(material["specular"])
                m.set_metallic(material["metallic"])
                m.set_base_color(material["base_color"])
                self.render_materials[material["name"]] = m

    def _load_articulations(self):
        self.articulations = {}
        if "articulations" not in self.level_config["layout"]:
            return
        for articulation_config in self.level_config["layout"]["articulations"]:
            if "urdf_file" in articulation_config:
                urdf = articulation_config["urdf_file"]
            else:
                urdf = download_partnet_mobility(
                    articulation_config["partnet_mobility_id"],
                    token=None,
                    directory=None,
                )

            loader = self._scene.create_urdf_loader()
            loader.scale = articulation_config.get("scale", 1)
            loader.fix_root_link = articulation_config.get("fix_base", True)

            config = {}
            if "surface_material" in articulation_config:
                config["material"] = self.physical_materials[articulation_config["surface_material"]]

            articulation = loader.load(urdf, config=config)
            if "initial_qpos" in articulation_config:
                articulation.set_qpos(articulation_config["initial_qpos"])
            articulation.set_root_pose(
                Pose(articulation_config["position"], articulation_config["rotation"])
            )
            articulation.set_name(articulation_config["name"])
            self.articulations[articulation_config["name"]] = {
                "articulation": articulation,
                "init_state": articulation.pack(),
            }

    def _load_actors(self):
        self.actors = {}
        if "rigid_bodies" in self.level_config["layout"]:
            for actor in self.level_config["layout"]["rigid_bodies"]:
                self._load_actor_from_config(actor)

    def _load_actor_from_config(self, actor):
        # special case for ground
        if actor["parts"] and actor["parts"][0]["type"] == "ground":
            shape = actor["parts"][0]
            a = self._scene.add_ground(
                shape["altitude"],
                shape["visual"],
                self.physical_materials[shape["surface_material"]]
                if "surface_material" in shape
                else None,
                self.render_materials[shape["render_material"]]
                if "render_material" in shape
                else None,
            )
            a.set_name(actor["name"])
            self.actors[actor["name"]] = {"actor": a, "init_state": a.pack()}
            return

        # all other actors
        builder = self._scene.create_actor_builder()
        is_kinematic = actor["kinematic"] if "kinematic" in actor else False
        if "mass" in actor:
            assert "inertia" in actor and "center_of_mass" in actor
            builder.set_mass_and_inertia(
                actor["mass"],
                Pose(
                    actor["center_of_mass"]["position"],
                    actor["center_of_mass"]["rotation"],
                ),
                actor["inertia"],
            )
        for shape in actor["parts"]:
            position = shape.get("position", [0, 0, 0])
            rotation = shape.get("rotation", [1, 0, 0, 0])
            assert "type" in shape
            if shape["type"] in ["box", "sphere", "capsule"]:
                if shape["collision"]:
                    shape_func = getattr(builder, "add_{}_collision".format(shape["type"]))
                    shape_func(
                        Pose(position, rotation),
                        shape["size"],
                        material=self.physical_materials[shape["material"]],
                        density=shape["physical_density"],
                    )
                if shape["visual"]:
                    visual_func = getattr(
                        builder, "add_{}_visual".format(shape["type"])
                    )
                    if "render_material" in shape:
                        render_mat = self.render_materials[shape["render_material"]]
                    else:
                        render_mat = self._renderer.create_material()
                        if "color" in shape:
                            render_mat.set_base_color(shape["color"])
                    visual_func(Pose(position, rotation), shape["size"], render_mat)
            elif shape["type"] == "mesh":
                if shape["collision"]:
                    builder.add_multiple_collisions_from_file(
                        shape["file"],
                        Pose(position, rotation),
                        scale=shape["scale"],
                        density=shape["physical_density"],
                    )
                if shape["visual"]:
                    builder.add_visual_from_file(
                        shape["file"], Pose(position, rotation), scale=shape["scale"]
                    )
            else:
                raise NotImplementedError

        a = builder.build(is_kinematic)
        a.set_name(actor["name"])
        a.set_pose(Pose(actor["position"], actor["rotation"]))
        self.actors[actor["name"]] = {"actor": a, "init_state": a.pack()}

    def _setup_objects(self):
        self.objects = {}
        for k, v in self.actors.items():
            self.objects[k] = v["actor"]
        for k, v in self.articulations.items():
            self.objects[k] = v["articulation"]

    def _load_agent(self):
        agent_config = self.level_config["agent"]
        if isinstance(agent_config, list):
            agents = []
            for config in agent_config:
                module_name, class_name = config["agent_class"].rsplit(".", 1)
                module = importlib.import_module(module_name)
                AgentClass = getattr(module, class_name)
                agents.append(AgentClass(self._engine, self._scene, config))
            self.agent = CombinedAgent(agents)
        else:
            module_name, class_name = agent_config["agent_class"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            AgentClass = getattr(module, class_name)
            self.agent = AgentClass(self._engine, self._scene, agent_config)


        if self.simulation_frequency % self.agent.control_frequency != 0:
            warnings.warn(
                "Simulation frequency does not divide agent control frequency. The number of simulation step per control step will be rounded"
            )
        self.n_simulation_per_control_step = self.simulation_frequency // self.agent.control_frequency

    def _load_custom(self):
        self.custom = self.level_config['custom'] if 'custom' in self.level_config else None

    def _observation_to_space(self, obs):
        if self.obs_mode == "state":
            low = np.full(obs.shape, -float("inf"), dtype=np.float32)
            high = np.full(obs.shape, float("inf"), dtype=np.float32)
            return spaces.Box(low, high, dtype=obs.dtype)
        elif self.obs_mode in ["rgbd", "pointcloud"]:
            agent_space = spaces.Box(
                -float("inf"),
                float("inf"),
                shape=obs["agent"].shape,
                dtype=obs["agent"].dtype,
            )
            ob_space = {}
            for camera_name, view in obs[self.obs_mode].items():
                ob_space[camera_name] = spaces.Box(
                    low=-float("inf"), high=-float("inf"), 
                    shape=view.shape, dtype=np.float32,
                ) # TODO: subsample or padding pointcloud
            # ob_space = [spaces.Box(
            #     low=-float("inf"), high=-float("inf"), 
            #     shape=view.shape, dtype=np.float32,
            # ) for view in obs[self.obs_mode]] # TODO: subsample or padding pointcloud
            return {
                "agent": agent_space.shape,
                self.obs_mode: ob_space,
            }
        else:
            raise Exception('Unknown obs mode.')

    def get_obs(self, **kwargs): # this is only used for default obs, if you want another obs, use render()
        if self.obs_mode == "state":
            return self.get_state(self.observer)
        elif self.obs_mode == "rgbd":
            return {
                "agent": self.agent.get_state(observer=self.observer),
                "additional_task_info": self.get_additional_task_info(obs_mode=self.obs_mode, observer=self.observer),
                "rgbd": self.render(mode='rgb_array', depth=True, camera_names=[self.observer], **kwargs),
            }
        elif self.obs_mode == "pointcloud":
            return {
                "agent": self.agent.get_state(observer=self.observer),
                "additional_task_info": self.get_additional_task_info(obs_mode=self.obs_mode, observer=self.observer),
                "pointcloud": self.render(mode="pointcloud", camera_names=[self.observer], **kwargs), # dict of partial point clouds
            }

    def get_state(self, observer):
        return np.concatenate([
            self.get_pad_object_state(observer=observer), 
            self.get_additional_task_info(obs_mode='state', observer=observer), 
            self.agent.get_state(observer=observer),
            ])

    def get_pad_object_state(self, observer):
        base_pos, base_quat, base_vel, base_ang_vel, qpos, qvel = self.get_object_state(observer=observer)
        k = len(qpos)
        pad_obj_internal_state = np.zeros(2 * self.obj_max_dof)
        pad_obj_internal_state[:k] = qpos
        pad_obj_internal_state[self.obj_max_dof : self.obj_max_dof+k] = qvel
        return np.concatenate([base_pos, base_quat, base_vel, base_ang_vel, pad_obj_internal_state])

    def get_object_state(self, observer):
        raise NotImplementedError()

    def get_articulation_state(self, art, observer):
        root_link = art.get_links()[0]
        base_pose = root_link.get_pose()
        base_vel = root_link.get_velocity()
        base_ang_vel = root_link.get_angular_velocity()
        qpos = art.get_qpos()
        qvel = art.get_qvel()
        if observer == 'world':
            return base_pose.p, base_pose.q, base_vel, base_ang_vel, qpos, qvel
        elif observer == 'robot':
            t_world_to_robot = self.agent.get_pose().to_transformation_matrix()
            t_robot_to_world = np.linalg.inv(t_world_to_robot)
            r, t = t_robot_to_world[:3,:3], t_robot_to_world[:3,3]
            return r @ base_pose.p + t, mat2quat( r @ quat2mat(base_pose.q) ), \
                r @ base_vel, r @ base_ang_vel, qpos, qvel
        else:
            raise NotImplementedError()

    def get_additional_task_info(self, obs_mode, observer):
        raise NotImplementedError()

    def get_custom_observation(self):
        raise NotImplementedError()

    def compute_reward(self, action):
        raise NotImplementedError()

    def set_state(self, state, observer='world'):
        raise NotImplementedError()

    def _eval(self):
        raise NotImplementedError()

    def _clip_and_scale_action(self, action): # from [-1, 1] to real action range
        action = np.clip(action, -1, 1)
        t = self.agent.action_range()
        action = 0.5 * (t.high - t.low) * action + 0.5 * (t.high + t.low) 
        return action

    def step(self, action):
        action = self._clip_and_scale_action(action)
        for __ in range(self.frame_skip):
            self.agent.set_action(action.copy()) # avoid action being changed
            for _ in range(self.n_simulation_per_control_step):
                self.agent.simulation_step()
                self._scene.step()
        if self.skip_reward:
            reward, info = 0, {}
        else:
            reward, info = self.compute_reward(action)
        obs = self.get_obs()
        info['eval_info'], done = self._eval()
        
        log_info(info)

        return obs, reward, done, info

    def mpc_step(self, state, action): # for mpc, avoid max_episode_steps constraint
        state_backup = self.get_state("world")
        self.set_state(state)
        action = self._clip_and_scale_action(action)
        for __ in range(self.frame_skip):
            self.agent.set_action(action.copy()) # avoid action being changed
            for _ in range(self.n_simulation_per_control_step):
                self.agent.simulation_step()
                self._scene.step()
        if self.skip_reward:
            reward, info = 0, {}
        else:
            reward, info = self.compute_reward(action)
        next_state = self.get_state("world")
        self.set_state(state_backup)
        return next_state, reward, info

    def accumulate_eval_results(self, flag_dict):
        eval_result_dict = {}
        for key, value in flag_dict.items():
            if value:
                self.keep_good_steps[key] += 1
            else:
                self.keep_good_steps[key] = 0
            eval_result_dict[key] = self.keep_good_steps[key] >= self.keep_good_steps_threshold
        return eval_result_dict, eval_result_dict['success']

    def render(self, mode="rgb_array", depth=False, normal=False, seg=False, camera_names=None):
        self._scene.update_render()
        if mode == "human":
            if self._viewer is None:
                self._viewer = Viewer(self._renderer)
                self._viewer.paused = False
                self._viewer.set_scene(self._scene)
                self._viewer.set_camera_xyz(-2, 0, 0.5)
                self._viewer.set_camera_rpy(0, -0.5, 0)
            self._viewer.render()
            return self._viewer
        else:
            if camera_names is None:
                cameras = self.cameras
            else:
                cameras = []
                for camera in self.cameras:
                    if camera.get_name() in camera_names:
                        cameras.append(camera)
            if mode == "rgb_array":
                imgs = {}
                for cam in cameras:
                    cam.take_picture()
                    img = cam.get_float_texture("Color")[:, :, :3]
                    if depth:
                        img = np.concatenate([img, cam.get_float_texture("GbufferDepth")[..., None]], -1)
                    if normal:
                        img = np.concatenate([img, cam.get_float_texture("Normal")], -1)
                    if seg:
                        seg = cam.get_uint32_texture("Segmentation")[..., 1]
                        seg = np.expand_dims(seg, axis=2)
                        img = np.concatenate([img, seg], -1)
                    imgs[cam.get_name()] = img
                return imgs
            elif mode == "pointcloud":
                pcds = {}
                for cam in cameras:
                    cam.take_picture()
                    img = cam.get_float_texture("Color")[:, :, :3]
                    pcd = cam.get_float_texture("Position")[:, :, :3]
                    seg = cam.get_uint32_texture("Segmentation")[..., 1]
                    mask = cam.get_float_texture("GbufferDepth") < 1
                    pcds[cam.get_name()] = np.concatenate([pcd[mask], img[mask], seg[mask][..., None]], -1)
                return pcds

    def close(self):
        if hasattr(self, '_viewer') and self._viewer:
            self._viewer.close()
        self._viewer = None
        self._scene = None

    def seed(self, seed=None):
        if seed is None:
            return self._main_seed
        else:
            self._setup_main_rng(seed=seed)

    def __del__(self):
        self.close()
    
