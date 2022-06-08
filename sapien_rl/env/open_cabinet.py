import numpy as np

from sapien.core import Pose, Articulation
import sapien.core as sp

from sapien_rl.env.sapien_env import SapienEnv
from sapien_rl.utils.geometry import sample_on_unit_sphere, rotation_between_vec
from sapien_rl.utils.contrib import compute_relative_vel, compute_dist2surface, pose_vec_distance, pose_corner_distance, apply_pose_to_points
from sapien_rl.utils.misc import sample_from_tuple_or_scalar
from scipy.spatial.transform import Rotation

import pathlib
_this_file = pathlib.Path(__file__).resolve()


class OpenCabinetEnv(SapienEnv):
    def __init__(self, fixed_target_link_id=None,
                 joint_friction=None, joint_damping=None, joint_stiffness=None,
                 *args, **kwargs):

        self.joint_friction = joint_friction
        self.joint_damping = joint_damping
        self.joint_stiffness = joint_stiffness

        self.gripper_init_dist = (0.15, 0.45)
        self.fixed_target_link_id = fixed_target_link_id # should only used for debugging
        super().__init__(
            _this_file.parent.joinpath(
                "../assets/config_files/open_cabinet.yml"
            ),
            *args,
            **kwargs,
        )

    def configure_env(self):
        self.obj_max_dof = 20

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.cabinet: Articulation = self.articulations["cabinet"]["articulation"]

        self._find_handles_from_articulation()
        self._close_all_parts()
        self._set_joint_physical_parameters()
        self._choose_part()
        self._place_gripper()
        self._ignore_collision()
        
        [[lmin, lmax]] = self.target_joint.get_limits()
        self.target_qpos = lmin + (lmax - lmin) * self.custom['open_extent']
        
        return self.get_obs()

    def _find_handles_from_articulation(self): 
        handles_info = {}
        o3d_info = {}
        gripper_info = {}

        from sapien_rl.utils.contrib import np2mesh, mesh2pcd, merge_mesh

        for link in self.articulations['cabinet']['articulation'].get_links():
            link_name = link.get_name()
            assert link_name not in handles_info
            handles_info[link_name] = []
            o3d_info[link_name] = []
            for visual_body in link.get_visual_bodies():
                if 'handle' not in visual_body.get_name():
                    continue
                for i in visual_body.get_render_shapes():
                    #from IPython import embed; embed()
                    vertices = apply_pose_to_points(i.mesh.vertices * visual_body.scale, visual_body.local_pose)
                    mesh = np2mesh(vertices, i.mesh.indices.reshape(-1, 3))
                    o3d_info[link_name].append(mesh)
                    handles_info[link_name].append((i.mesh.vertices * visual_body.scale, i.mesh.indices, visual_body.local_pose))
            if len(handles_info[link_name]) == 0:
                handles_info.pop(link_name)
                o3d_info.pop(link_name)

        for link in self.articulations['cabinet']['articulation'].get_links():
            link_name = link.get_name()
            if link_name not in o3d_info:
                continue
            mesh = merge_mesh(o3d_info[link_name])
            pcd = mesh2pcd(mesh, 50)
            o3d_info[link_name] = (link, mesh, pcd)
            #o3d.visualization.draw_geometries(])
        
        for link in self.agent.robot.get_links():
            link_name = link.get_name()
            if link_name not in ['panda_leftfinger', 'panda_rightfinger']:
                continue
            assert link_name not in gripper_info
            gripper_info[link_name] = []
            for visual_body in link.get_visual_bodies():
                for i in visual_body.get_render_shapes():
                    vertices = apply_pose_to_points(i.mesh.vertices * visual_body.scale, visual_body.local_pose)
                    mesh = np2mesh(vertices, i.mesh.indices.reshape(-1, 3))
                    gripper_info[link_name].append(mesh)
            if gripper_info[link_name] == 0:
                gripper_info.pop(link_name)
                continue
            mesh = merge_mesh(gripper_info[link_name])
            pcd = mesh2pcd(mesh, 50)
            gripper_info[link_name] = (link, mesh, pcd)
            #o3d.visualization.draw_geometries([gripper_info[link_name], ])

        self.gripper_info = gripper_info

        self.handles_info = handles_info
        self.o3d_info = o3d_info
        assert len(self.handles_info.keys()) > 0
    
    def _close_all_parts(self):
        qpos = []
        for joint in self.cabinet.get_active_joints():
            [[lmin, lmax]] = joint.get_limits()
            if lmin == -np.inf or lmax == np.inf:
                #print(self.level_config["layout"]["articulations"])
                raise Exception('This object has an inf limit joint.')
            qpos.append(lmin)
        self.cabinet.set_qpos(np.array(qpos))

    def _choose_part(self):
        links, joints = [], []
        for link, joint in zip(self.cabinet.get_links(), self.cabinet.get_joints()):
            if (joint.type == "revolute" or joint.type == "prismatic") and link.get_name() in self.handles_info:
                links.append(link)
                joints.append(joint)

        if self.fixed_target_link_id is not None:
            self.target_index = self.fixed_target_link_id % len(joints)
        else:
            self.target_index = self._level_rng.choice(len(joints)) # only sample revolute/prismatic joints
        self.target_link = links[self.target_index]
        self.target_link_name = self.target_link.get_name()
        self.target_joint = joints[self.target_index]
        self.target_index_in_active_joints = self.cabinet.get_active_joints().index(self.target_joint)
        self.target_indicator = np.zeros(self.obj_max_dof)
        self.target_indicator[self.target_index_in_active_joints] = 1
        #print(self.target_link_name)

    def get_additional_task_info(self, obs_mode, observer='world'):
        if obs_mode == 'state':
            return self.target_indicator
        else:
            return np.array([])

    def perturb_gripper(self):
        qpos = self.agent.robot.get_qpos()
        qpos[:3] += np.random.randn(3) * 0.03 
        qpos[3:6] += np.random.randn(3) * 0.3
        self.agent.set_state({
            'qpos': qpos,
        }, observer='world', by_dict=True)

    def _place_gripper(self):
        # shapes = self.target_link.get_collision_shapes()
        # mins = np.ones(3) * np.inf
        # maxs = -np.ones(3) * np.inf
        # for shape in shapes:
        #     assert shape.type == "convex_mesh"
        #     mat = (self.target_link.pose * shape.pose).to_transformation_matrix()
        #     scale = shape.convex_mesh_geometry.scale
        #     vertices = shape.convex_mesh_geometry.vertices
        #     world_vertices = (vertices * scale) @ (mat[:3, :3].T) + mat[:3, 3]
        #     mins = np.minimum(mins, world_vertices.min(0))
        #     maxs = np.maximum(maxs, world_vertices.max(0))

        shapes = self.handles_info[self.target_link.name]
        mins = np.ones(3) * np.inf
        maxs = -np.ones(3) * np.inf
        for scaled_vertices, _, shape_pose in shapes:
            mat = (self.target_link.pose * shape_pose).to_transformation_matrix()
            world_vertices = scaled_vertices @ (mat[:3, :3].T) + mat[:3, 3]
            mins = np.minimum(mins, world_vertices.min(0))
            maxs = np.maximum(maxs, world_vertices.max(0))

        handle_center = 0.5 * (mins + maxs)

        dist = self._level_rng.uniform(low=self.gripper_init_dist[0], high=self.gripper_init_dist[1])
        delta = sample_on_unit_sphere(self._level_rng) * dist
        delta[0] = -abs(delta[0])
        ref_point_on_handle = np.array([mins[0]-0.05, handle_center[1], handle_center[2] ])
        gripper_pos = ref_point_on_handle + delta

        qpos = self.agent.robot.get_qpos()
        qpos[:3] = gripper_pos

        xyz_joint_rotation = self._level_rng.uniform(
            low=[-1.5707963267948966, -1.5707963267948966, 0],
            high=[1.5707963267948966, -1.5707963267948966, 0,],
        ) # face to positive x, and random roation along x
        # xyz_joint_rotation = np.array([0.5, -1.5707963267948966, 0,])
        current_gripper_rot_mat = Rotation.from_euler('ZYX', xyz_joint_rotation).as_matrix() # (3, ), intrinsic

        rotation_from_x_axis_to_vec_to_handle = rotation_between_vec(np.array([1,0,0]), handle_center - gripper_pos)
        current_gripper_rot_mat = current_gripper_rot_mat @ rotation_from_x_axis_to_vec_to_handle.as_matrix()  # align x-axis with vector points to handle

        orientation_z_half_sphere = sample_on_unit_sphere(self._level_rng)
        orientation_z_half_sphere[2] = -abs(orientation_z_half_sphere[2])
        perturb_rotation = rotation_between_vec(np.array([0,0,-1]), orientation_z_half_sphere) # negative z points outside of gripper
        # print('perturb angle:', np.linalg.norm(perturb_rotation.as_rotvec()) / np.pi * 180)
        current_gripper_rot_mat = current_gripper_rot_mat @ perturb_rotation.as_matrix() # perturb a bit, <90 deg with x-axis

        qpos[3:6] = Rotation.from_matrix(current_gripper_rot_mat).as_euler('ZYX') # (3, ), intrinsic
        
        #qpos[3:6] = 0
        #qpos[4] = -np.pi / 2
        #qpos[3] = np.pi / 2

        # self.agent.robot.set_qpos(qpos)
        self.agent.set_state({
            'qpos': qpos,
        }, observer='world', by_dict=True)

    def _ignore_collision(self):
        '''ignore collision among all movable links'''
        cabinet = self.articulations["cabinet"]["articulation"]
        for joint, link in zip(cabinet.get_joints(), cabinet.get_links()):
            if joint.type in ["revolute", "prismatic"]:
                shapes = link.get_collision_shapes()
                for s in shapes:
                    g0,g1,g2,g3 = s.get_collision_groups()
                    s.set_collision_groups(g0,g1,g2|1<<31,g3)

    def get_object_state(self, observer):
        return self.get_articulation_state(art=self.cabinet, observer=observer)

    def get_state(self, observer):
        return super().get_state(observer=observer)

    def set_state(self, state, observer='world'):
        if observer == 'world':
            # cabinet is fixed, so we dont need to set base pos / vel
            state = state[13:]
            self.cabinet.set_qpos(state[0:self.cabinet.dof])
            self.cabinet.set_qvel(state[self.obj_max_dof:self.obj_max_dof+self.cabinet.dof])

            # set robot state
            task_info_len = len(self.get_additional_task_info('state', 'world'))
            self.agent.set_state(state[2*self.obj_max_dof + task_info_len:])

            return self.get_obs()
        else:
            raise NotImplementedError('set_state only supports observer=world')

    def _set_joint_physical_parameters(self):
        for joint in self.cabinet.get_joints():
            # print('damping:', joint.damping)
            # print('friction:', joint.friction)
            # print('stiffness:', joint.stiffness)
            if self.joint_friction is not None:
                joint.set_friction(sample_from_tuple_or_scalar(self._level_rng, self.joint_friction))
            if self.joint_damping is not None:
                joint.set_drive_property(stiffness=0, 
                                        damping=sample_from_tuple_or_scalar(self._level_rng, self.joint_damping),
                                        force_limit=3.4028234663852886e+38)
        # print('------------')

    def get_custom_observation(self):
        ee_coords = np.array(self.agent.get_ee_coords())
        ee_vels = np.array(self.agent.get_ee_vels())
        ee_q = self.agent.get_ee_orientation()

        actor = self.target_link
        target_handle_pcd = self.o3d_info[self.target_link_name][-1]
        target_handle_points = apply_pose_to_points(np.asarray(target_handle_pcd.points), actor.get_pose())
        handle_position = np.mean(target_handle_points, axis=0)
        # print(handle_position.shape)

        object_pose = self.articulations["cabinet"]["articulation"].get_qpos() # pos of active joints
        object_vel = self.articulations["cabinet"]["articulation"].get_qvel()

        ee_relative_coords = np.concatenate(ee_coords - handle_position, axis=0)

        dist_ee_actor = np.array(np.sqrt(((ee_coords[:, None] - target_handle_points[None]) ** 2).sum(-1)).min(-1).mean()).reshape(1)

        obj = self.articulations["cabinet"]["articulation"]
        finish_sign = np.array(obj.get_qpos()[self.target_index_in_active_joints] >= self.target_qpos).reshape(1)

        # for x in [ee_relative_coords, np.concatenate(ee_vels), ee_q, handle_position, object_pose, object_vel,
        #                       dist_ee_actor, finish_sign]:
        #     print(x.shape)
        ret = np.concatenate([ee_relative_coords, np.concatenate(ee_vels), ee_q, handle_position, object_pose, object_vel,
                              dist_ee_actor, finish_sign], axis=0)
        return ret

    def compute_reward(self, action, state=None):
        reward = 0

        info_dict = {}
        info_dict['state_info'] = {
            'gripper_pos': self.agent.robot.get_qpos()[:3].tolist(),
            'qpos': self.cabinet.get_qpos()[self.target_index_in_active_joints],
            'target_qpos': self.target_qpos,
            'link_vel': np.linalg.norm(self.target_link.get_velocity()),
            'link_ang_vel': np.linalg.norm(self.target_link.get_angular_velocity()),
        }

        state_info = info_dict['state_info']

        DIST_THRESHOLD = 0.025
        OPEN_ENOUGH_REWARD_CONST = 5
        CLOSE_CONST = 3

        open_enough = self.cabinet.get_qpos(
        )[self.target_index_in_active_joints] >= self.target_qpos

        link_velocity = state_info['link_vel']
        link_angular_velocity = state_info['link_ang_vel']

        if open_enough:
            reward = OPEN_ENOUGH_REWARD_CONST - link_velocity - link_angular_velocity
            return reward, info_dict

        custom_obs = self.get_custom_observation()

        ee_relative_coords = custom_obs[:3] # First arm
        object_pose = custom_obs[19]
        target_pose = state_info['target_qpos']

        dist = np.linalg.norm(ee_relative_coords) # Distance from handle
        if dist > DIST_THRESHOLD:
            reward = -dist
        else:
            reward = CLOSE_CONST + object_pose - target_pose

        return reward, info_dict


    def get_obs(self):
        obs = super().get_obs(seg=True)
        if self.obs_mode == "pointcloud" or self.obs_mode == 'rgbd':
            views = obs[self.obs_mode]
            for cam_name, view in views.items():
                mask = ( view[..., -1] == self.target_link.get_id() )
                view[..., -1] = 0
                view[..., -1][mask] = 1
        return obs

    def _eval(self):
        flag_dict = {
            "object_static": (np.linalg.norm(self.target_link.get_velocity()) <= 0.05)
                        and (np.linalg.norm(self.target_link.get_angular_velocity()) <= 1),
            "open_enough": self.cabinet.get_qpos()[self.target_index_in_active_joints] >= self.target_qpos,
        }
        flag_dict["success"] = all(flag_dict.values())
        return self.accumulate_eval_results(flag_dict)

    @property
    def num_links(self):
        links, joints = [], []
        for link, joint in zip(self.cabinet.get_links(), self.cabinet.get_joints()):
            if (joint.type == sp.REVOLUTE or joint.type == sp.PRISMATIC) and link.get_name() in self.handles_info:
                links.append(link)
                joints.append(joint)
        return len(links)


