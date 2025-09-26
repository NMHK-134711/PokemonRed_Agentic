import os
import json

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from einops import rearrange, reduce

def merge_dicts(dicts):
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)): 
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return mean_dict, distrib_dict

class TensorboardCallback(BaseCallback):

    def __init__(self, log_dir=None, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self):
        pass

    def _on_step(self) -> bool:
        
        # 매 100 step마다 각 환경별 누적 reward와 화면 로그
        if self.n_calls % 100 == 0:
            # 각 환경의 cumulative_reward 가져오기
            all_cum_rewards = self.training_env.get_attr("cumulative_reward")
            for i, cum_reward in enumerate(all_cum_rewards):
                self.logger.record(f"debug/cumulative_reward_env{i}", cum_reward)
            
            all_died_counts = self.training_env.get_attr("died_count")
            for i, count in enumerate(all_died_counts):
                self.logger.record(f"debug/death_count_env{i}", count)

            # 각 환경의 recent_screens 가져와 Image로 로그
            all_screens = self.training_env.get_attr("recent_screens")
            for i, screens in enumerate(all_screens):
                if screens is not None and screens.size > 0:
                    self.logger.record(f"debug/screen_env{i}", Image(screens, "HWC"), exclude=("stdout", "log", "json", "csv"))

        if self.training_env.env_method("check_if_done", indices=[0])[0]:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos]
            mean_infos, distributions = merge_dicts(all_final_infos)
            # TODO log distributions, and total return
            for key, val in mean_infos.items():
                self.logger.record(f"env_stats/{key}", val)

            for key, distrib in distributions.items():
                # self.writer.add_histogram 대신 self.logger.record 사용
                self.logger.record(f"env_stats_distribs/{key}", distrib) # <--- 수정
                self.logger.record(f"env_stats_max/{key}", max(distrib))
                        
            explore_map = np.array(self.training_env.get_attr("explore_map"))
            map_sum = reduce(explore_map, "f h w -> h w", "max")
            self.logger.record("trajectory/explore_sum", Image(map_sum, "HW"), exclude=("stdout", "log", "json", "csv"))

            map_row = rearrange(explore_map, "(r f) h w -> (r h) (f w)", r=2)
            self.logger.record("trajectory/explore_map", Image(map_row, "HW"), exclude=("stdout", "log", "json", "csv"))

            list_of_flag_dicts = self.training_env.get_attr("current_event_flags_set")
            merged_flags = {k: v for d in list_of_flag_dicts for k, v in d.items()}
            self.logger.record("trajectory/all_flags", json.dumps(merged_flags))

        return True