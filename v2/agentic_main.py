import time
import os
import torch.multiprocessing as mp
from pathlib import Path
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure as configure_logger

from red_gym_env_agentic import RedGymEnvAgentic
from llm_planner import LLMPlanner
from skill_library_red import AVAILABLE_SKILLS_RED
from tensorboard_callback import TensorboardCallback
from custom_policy import CombinedExtractor

mp.set_start_method('spawn', force=True)

# --- 환경 생성 함수 ---
def make_env(rank, env_conf, seed=0):
    # 이제 custom_init_state 인자는 필요 없으므로 삭제합니다.
    def _init():
        env = RedGymEnvAgentic(env_conf)
        env.reset(seed=(seed + rank))
        return env
    return _init

# --- 메인 실행 ---
if __name__ == '__main__':
    # --- 경로 설정 ---
    current_time = int(time.time())
    logs_path = Path("logs") / f"train_log_{current_time}"
    model_path = Path("models") / f"train_model_{current_time}"
    logs_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    print(f"로그 저장 경로: {logs_path}")
    print(f"모델 저장 경로: {model_path}")

    # --- 환경 및 Agentic 구성 요소 초기화 ---
    num_cpu = 10
    ep_length = 2048 * 20

    env_config = {
        'headless': True, 
        'save_final_state': False, 
        'early_stop': False,
        'action_freq': 24, 
        'init_state': '../initial_red.state', 
        'max_steps': ep_length,
        'print_rewards': True, 
        'save_video': False, 
        'fast_video': True, 
        'session_path': logs_path,
        'gb_path': '../PokemonRed.gb', 
        'debug': False, 
        'reward_scale': 1.0, 
        'explore_weight': 0.2
    }
    
    # make_env 함수 호출 시 custom_init_state를 전달하지 않습니다.
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    # VecTransposeImage 래퍼는 그대로 사용합니다.
    env = VecTransposeImage(env)


    print("--- Agentic AI 구성 요소를 초기화합니다 ---")
    llm_planner = LLMPlanner()

    # --- 저수준 액터(PPO) 모델 초기화 ---
    LOAD_MODEL = False
    MODEL_TO_LOAD_PATH = "models/best_model/final_model.zip"

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
    )

    if LOAD_MODEL:
        print(f"저장된 모델을 불러옵니다: {MODEL_TO_LOAD_PATH}")
        model = RecurrentPPO.load(
            MODEL_TO_LOAD_PATH,
            env=env,
            custom_objects={"policy_kwargs": policy_kwargs}
        )
        model.set_logger(configure_logger(verbose=1, tensorboard_log=logs_path))
    else:
        print("새로운 PPO 모델을 생성합니다.")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=2048,
            batch_size=128,
            n_epochs=4,
            gamma=0.999,
            tensorboard_log=logs_path,
        )

    callbacks = [
        TensorboardCallback(log_dir=logs_path, verbose=1),
        CheckpointCallback(save_freq=10000, save_path=str(model_path), name_prefix="ppo_model")
    ]

    total_steps = 0
    segment_count = 0
    steps_per_segment = model.n_steps * num_cpu 

    # [제거] 초기 상태 동기화는 각 환경이 reset에서 스스로 수행하므로 필요 없음

    while total_steps < 10_000_000:
        segment_count += 1
        print(f"\n{'='*20} 세그먼트 {segment_count} 시작 {'='*20}")

        # 1. 각 환경이 스스로 목표를 갱신하도록 명령
        env.env_method('check_and_advance_task')
        
        # 2. 각 환경으로부터 '개별 목표'와 '게임 상태'를 모두 가져옴
        all_game_states = env.env_method('get_structured_state')
        all_task_descriptions = env.env_method('get_task_description')

        print("--- 각 에이전트의 현재 목표 ---")
        for i, desc in enumerate(all_task_descriptions):
            print(f"  - 에이전트 {i}: {desc.splitlines()[1]}") # 구체적인 목표만 출력
        print("-----------------------------")
        
        # 3. LLM에 모든 정보를 전달하여 개별 스킬들을 결정
        print("LLM 플래너를 호출하여 각 에이전트의 다음 목표 스킬을 결정합니다...")
        chosen_skills = llm_planner.choose_next_skill_batch(
            all_game_states, 
            all_task_descriptions, 
            AVAILABLE_SKILLS_RED
        )
        
        # 4. 결정된 개별 스킬을 각 환경에 설정
        for i, skill in enumerate(chosen_skills):
            env.env_method('set_current_skill', skill, indices=[i])

        # 5. 저수준 액터 학습 수행
        print(f"저수준 PPO 액터가 {steps_per_segment} 스텝 동안 목표 스킬을 수행하며 학습합니다...")
        model.learn(
            total_timesteps=steps_per_segment, 
            reset_num_timesteps=False,
            callback=callbacks,
        )
        
        total_steps += steps_per_segment

    print("학습이 완료되었습니다.")
    print("최종 평가를 진행하여 최고 환경의 상태를 저장합니다...")

    # 최종 평가 및 최고 상태 저장 로직은 그대로 유용하게 사용할 수 있습니다.
    final_stats = env.env_method('get_agent_priority_stats')
    best_final_stats = max(final_stats)
    best_final_agent_idx = final_stats.index(best_final_stats)
    print(f"최고 성과 에이전트: {best_final_agent_idx}, 스탯: {best_final_stats}")

    best_state_data = env.env_method('get_pyboy_state', indices=[best_final_agent_idx])[0]
    save_path = model_path / "best_env_final.state"
    with open(save_path, "wb") as f:
        f.write(best_state_data)
    print(f"최고 환경의 상태를 '{save_path}'에 저장했습니다.")

    model.save(model_path / "final_model.zip")
    env.close()