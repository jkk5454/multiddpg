from softgym.envs.cloth_move import ClothMoveEnv

from collections import OrderedDict

env_arg_dict = {
    'ClothMove': {'observation_mode': 'cam_rgb',
                  'action_mode': 'picker',
                  'num_picker': 2,
                  'render': True,
                  'headless': True,
                  'horizon': 160,
                  'action_repeat': 8,
                  'render_mode': 'cloth',
                  'num_variations': 1000,
                  'use_cached_states': True,
                  'deterministic': False},
}

SOFTGYM_ENVS = OrderedDict({
    'ClothMove': ClothMoveEnv,
})
