import numpy as np
from gym import register
import pathlib
import yaml


################################################################

_this_file = pathlib.Path(__file__).resolve()
cabinet_model_file = _this_file.parent.joinpath(
    "../assets/config_files/cabinet_models.yml")
path = pathlib.Path(cabinet_model_file).resolve()
with path.open("r") as f:
    raw_yaml = yaml.load(f, Loader=yaml.SafeLoader)
cabinet_id_list = list(raw_yaml.keys())


# ----------------------------- v4
joint_friction = (0.05, 0.15)
joint_damping = (5, 20)

for obs_mode in ['pointcloud', 'state', 'rgbd']:
    register(
        id='OpenCabinet_{:s}-v4'.format(obs_mode),
        entry_point='sapien_rl.env.open_cabinet:OpenCabinetEnv',
        max_episode_steps=150,
        kwargs={'frame_skip':5, 'obs_mode': obs_mode, 
                'joint_friction': joint_friction, 'joint_damping': joint_damping, }
    )

    for cabinet_id in cabinet_id_list:
        if cabinet_id != '45267': continue
        register(
            id='OpenCabinet_{:s}_{:s}-v4'.format(obs_mode, cabinet_id),
            entry_point='sapien_rl.env.open_cabinet:OpenCabinetEnv',
            max_episode_steps=150,
            kwargs={'frame_skip':5, 'obs_mode': obs_mode, 
                'variant_config': {"partnet_mobility_id": cabinet_id},
                'joint_friction': joint_friction, 'joint_damping': joint_damping,
            }
        )

        for fixed_target_link_id in range(5):
            register(
                id='OpenCabinet_{:s}_{:s}_link_{:d}-v4'.format(obs_mode, cabinet_id, fixed_target_link_id),
                entry_point='sapien_rl.env.open_cabinet:OpenCabinetEnv',
                max_episode_steps=150,
                kwargs={'frame_skip':5, 'obs_mode': obs_mode, 
                    'variant_config': {"partnet_mobility_id": cabinet_id},
                    'fixed_target_link_id': fixed_target_link_id,
                    'joint_friction': joint_friction, 'joint_damping': joint_damping,
                }
            )
