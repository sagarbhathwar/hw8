import numpy as np
from gym import spaces
from sapien.core import Pose, Engine, Scene, Articulation
from sapien_rl.agent.controllers import LPFilter, PIDController, VelocityController, PositionController
import yaml
import transforms3d
from sapien_rl.utils.geometry import rotate_2d_vec_by_angle

class Agent:
    def __init__(self, engine: Engine, scene: Scene, config):
        if type(config) == str:
            with open(config, 'r') as f:
                config = yaml.safe_load(f)['agent']

        self.config = config
        self._engine = engine
        self._scene = scene

        self.control_frequency = config["control_frequency"]

        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = config["fix_base"]
        loader.scale = config["scale"]

        self._physical_materials = {}
        if config["surface_materials"]:
            for mat in config["surface_materials"]:
                self._physical_materials[mat["name"]] = scene.create_physical_material(
                    mat["static_friction"], mat["dynamic_friction"], mat["restitution"]
                )

        urdf_config = {"link": {}}
        for link in self.config["links"]:
            link_props = {}
            if "surface_material" in link:
                link_props["surface_material"] = self._physical_materials[link["surface_material"]]
            if "patch_radius" in link:
                link_props["patch_radius"] = link["patch_radius"]
            if "min_patch_radius" in link:
                link_props["min_patch_radius"] = link["min_patch_radius"]
            urdf_config["link"][link['name']] = link_props

        self.robot = loader.load(config["urdf_file"], urdf_config)
        self.robot.set_name(self.config["name"])

        self.active_joints = self.robot.get_active_joints()

        self.balance_passive_force = config["balance_passive_force"]

        assert (
            len(self.config["initial_qpos"]) == self.robot.dof
        ), "initial_qpos does not match robot DOF"

        self.robot.set_qpos(self.config["initial_qpos"])
        self.robot.set_root_pose(Pose(self.config["base_position"], self.config["base_rotation"]))
        
        self._init_state = self.robot.pack()

        self.all_joint_indices = [
            [x.name for x in self.robot.get_active_joints()].index(name)
            for name in self.config["all_joints"]
        ]
        self.controllable_joint_indices = [
            [x.name for x in self.robot.get_active_joints()].index(name)
            for name in self.config["controllable_joints"]
        ]

        name2pxjoint = dict((j.get_name(), j) for j in self.robot.get_joints())
        name2config_joint = dict((j["name"], j) for j in config["joints"])

        for joint in config["joints"]:
            assert (
                joint["name"] in name2pxjoint
            ), "Unrecognized name in joint configurations"
            j = name2pxjoint[joint["name"]]

            stiffness = joint.get("stiffness", 0)
            damping = joint["damping"]
            friction = joint["friction"]
            j.set_drive_property(stiffness, damping)
            j.set_friction(friction)

        controllers = []
        all_action_range = []
        for name in self.config["controllable_joints"]:
            assert (
                name in name2config_joint
            ), "Controllable joints properties must be configured"
            joint = name2config_joint[name]
            action_type = joint["action_type"]
            action_range = joint["action_range"]

            all_action_range.append(action_range)

            velocity_filter = None
            if "velocity_filter" in joint:
                velocity_filter = LPFilter(
                    self.control_frequency, joint["velocity_filter"]["cutoff_frequency"]
                )
            if action_type == "velocity":
                controller = VelocityController(velocity_filter)
            elif action_type == "position":
                kp = joint["velocity_pid"]["kp"]
                ki = joint["velocity_pid"]["ki"]
                kd = joint["velocity_pid"]["kd"]
                limit = joint["velocity_pid"]["limit"]
                controller = PositionController(
                    PIDController(kp, ki, kd, self.control_frequency, limit),
                    velocity_filter,
                )
            else:
                raise RuntimeError("Only velocity or position are valid action types")
            controllers.append(controller)

        self.controllers = controllers
        all_action_range = np.array(all_action_range, dtype=np.float32)
        self._action_range = spaces.Box(all_action_range[:, 0], all_action_range[:, 1])

    def action_range(self):
        return self._action_range

    def set_action(self, action: np.ndarray):
        assert action.shape == self._action_range.shape

        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()

        for j_idx, controller, target in zip(
            self.controllable_joint_indices, self.controllers, action
        ):
            # import pdb; pdb.set_trace()
            if type(controller) == PositionController:
                output = controller.control(qpos[j_idx], target)
            elif type(controller) == VelocityController:
                output = controller.control(qvel[j_idx], target)
            else:
                raise Exception("this should not happen, please report it")
            self.active_joints[j_idx].set_drive_velocity_target(output)
            # import pdb; pdb.set_trace()

    def simulation_step(self):
        if self.balance_passive_force:
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True, external=False
            )
            self.robot.set_qf(qf)

    def get_state(self, observer='world', by_dict=False):
        if observer == 'world':
            qpos = self.robot.get_qpos()[self.all_joint_indices]
            qvel = self.robot.get_qvel()[self.all_joint_indices]
            controller_state = []
            for controller in self.controllers:
                if type(controller) == PositionController:
                    n = controller.velocity_pid._prev_err is not None
                    controller_state.append(n)
                    if n:
                        controller_state.append(controller.velocity_pid._prev_err)
                    else:
                        controller_state.append(0)
                    controller_state.append(controller.velocity_pid._cum_err)
                    controller_state.append(controller.lp_filter.y)
                elif type(controller) == VelocityController:
                    controller_state.append(controller.lp_filter.y)
            if by_dict:
                return {
                    'qpos': qpos,
                    'qvel': qvel,
                    'controller_state': np.array(controller_state),
                }
            else:
                return np.concatenate([qpos, qvel, controller_state])
        else:
            raise NotImplementedError()
    
    def set_state(self, state, observer='world', by_dict=False):
        if observer == 'world':
            if not by_dict:
                state_dict = {
                    'qpos': state[:self.robot.dof],
                    'qvel': state[self.robot.dof:2*self.robot.dof],
                    'contronller_state': state[2*self.robot.dof:],
                }
            else:
                state_dict = state
            if 'qpos' in state_dict:
                qpos = np.zeros(self.robot.dof)
                qpos[self.all_joint_indices] = state_dict['qpos']
                self.robot.set_qpos(qpos)
            if 'qvel' in state_dict:
                qvel = np.zeros(self.robot.dof)
                qvel[self.all_joint_indices] = state_dict['qvel']
                self.robot.set_qvel(qvel)
            if 'contronller_state' in state_dict:
                # idx = 2*self.robot.dof
                state = state_dict['contronller_state']
                idx = 0
                for controller in self.controllers:
                    if type(controller) == PositionController:
                        if state[idx]:
                            controller.velocity_pid._prev_err = state[idx+1]
                        else:
                            controller.velocity_pid._prev_err = None
                        controller.velocity_pid._cum_err = state[idx+2]
                        controller.lp_filter.y = state[idx+3]
                        idx = idx + 4
                    elif type(controller) == VelocityController:
                        controller.lp_filter.y = state[idx]
                        idx = idx + 1
        else:
            raise NotImplementedError()

    def reset(self):
        self.robot.unpack(self._init_state)

class FloatingPandaAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "floating"

    def get_ee_orientation(self):
        return self.robot.get_pose().q

    def get_ee_coords(self):
        finger_tips = [
            self.robot.get_joints()[-1]
            .get_global_pose()
            .transform(Pose([0, 0.04, 0]))
            .p,
            self.robot.get_joints()[-2]
            .get_global_pose()
            .transform(Pose([0, -0.04, 0]))
            .p,
        ]
        return np.array(finger_tips)

    def get_ee_vels(self):
        finger_vels = [
            self.robot.get_links()[-1].get_velocity(),
            self.robot.get_links()[-2].get_velocity(),
        ]
        return np.array(finger_vels)
    
    def randomize(self):
        rand_coord = np.random.uniform(0, 1, size=3)
        self.robot.set_pose(Pose(rand_coord))

    # RMOVE!!!
    def get_ee_coords_sample(self):
        l = 0.03
        r = 0.05
        ret = []
        for i in range(10):
            x = (l * i + (4 - i) * r) / 4
            finger_tips = [
                self.robot.get_joints()[-1]
                .get_global_pose()
                .transform(Pose([0, x, 0]))
                .p,
                self.robot.get_joints()[-2]
                .get_global_pose()
                .transform(Pose([0, -x, 0]))
                .p,
            ]
            ret.append(finger_tips)
        return np.array(ret).transpose((1, 0, 2))



