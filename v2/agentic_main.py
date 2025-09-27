# agentic_main.py
import time
import os
import numpy as np
from pathlib import Path
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure as configure_logger

from red_gym_env_agentic import RedGymEnvAgentic
from llm_planner import LLMPlanner
from task_manager_red import TaskManagerRed
from skill_library_red import AVAILABLE_SKILLS_RED
from tensorboard_callback import TensorboardCallback
from custom_policy import CombinedExtractor

# --- 환경 생성 함수 ---
def make_env(rank, env_conf, seed=0, custom_init_state=None):
    # custom_init_state가 제공되면, env_conf의 기본값을 덮어씁니다.
    if custom_init_state:
        env_conf['init_state'] = custom_init_state
    
    def _init():
        env = RedGymEnvAgentic(env_conf)
        env.reset(seed=(seed + rank))
        return env
    return _init

# --- 메인 실행 ---
if __name__ == '__main__':
    # --- 경로 설정 ---
    # 1. 현재 시간을 기준으로 고유한 ID 생성 (로그와 모델 폴더에 공유)
    current_time = int(time.time())

    # 2. 새로운 로그 및 모델 저장 경로 정의
    logs_path = Path("logs") / f"train_log_{current_time}"
    model_path = Path("models") / f"train_model_{current_time}"

    # 3. 해당 폴더들이 존재하지 않으면 생성
    logs_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    print(f"로그 저장 경로: {logs_path}")
    print(f"모델 저장 경로: {model_path}")

    # --- 환경 및 Agentic 구성 요소 초기화 ---
    num_cpu = 8  # 병렬 환경 수
    ep_length = 2048 * 10

    gb_path = os.path.abspath('../PokemonRed.gb')
    init_state_path = os.path.abspath('../has_pokedex.state') # 또는 initial_red.state
    
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
    
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    print("--- Agentic AI 구성 요소를 초기화합니다 ---")
    llm_planner = LLMPlanner()
    task_manager = TaskManagerRed(plan_path="pokemon_red_plan.json")

    # --- 저수준 액터(PPO) 모델 초기화 ---
    LOAD_MODEL = False  # True로 설정하면 아래 경로의 모델을 불러옵니다.
    MODEL_TO_LOAD_PATH = "models/best_model/final_model.zip" # 불러올 모델의 경로

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
    )

    if LOAD_MODEL:
        print(f"저장된 모델을 불러옵니다: {MODEL_TO_LOAD_PATH}")
        model = RecurrentPPO.load(
            MODEL_TO_LOAD_PATH,
            env=env,
            # load 할 때도 policy_kwargs 등을 명시해주는 것이 좋습니다.
            custom_objects={"policy_kwargs": policy_kwargs}
        )
        # 불러온 모델에 새로운 로그 경로를 설정합니다.
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

    # --- 학습 루프 ---
    total_steps = 0
    segment_count = 0
    steps_per_segment = 2048 * num_cpu  # LLM이 다음 스킬을 결정하는 주기

    # 초기 상태와 태스크 동기화
    initial_state = env.env_method('get_structured_state', indices=[0])[0]
    task_manager.sync_with_initial_state(initial_state)

    while total_steps < 10_000_000:
        segment_count += 1
        print(f"\n{'='*20} 세그먼트 {segment_count} 시작 {'='*20}")

        # 1. 모든 에이전트의 우선순위 스탯 가져오기
        all_stats = env.env_method('get_agent_priority_stats')
        
        print("--- 모든 에이전트 평가 결과 ---")
        for i, stats in enumerate(all_stats):
            print(f"  - 에이전트 {i}: {stats}")
        print("-----------------------------")

        # 2. 최고 에이전트 선정 (튜플 비교는 자동으로 우선순위 처리)
        best_stats = max(all_stats)
        # 가장 큰 튜플을 가진 에이전트의 인덱스(번호)를 찾습니다.
        best_agent_idx = all_stats.index(best_stats)
        
        print(f"에이전트 평가 완료. 최고 에이전트: {best_agent_idx}, 스탯: {best_stats}")

        # 3. 복제 건너뛰기 조건 확인
        # 모든 에이전트의 스탯이 최고 에이전트의 스탯과 동일한지 확인
        is_all_same = all(stats == best_stats for stats in all_stats)
        
        if is_all_same:
            print("모든 에이전트의 상태가 동일하여 복제를 건너뜁니다.")
        else:
            print(f"에이전트 {best_agent_idx}의 상태를 복제합니다...")
            
            # --- 새로운 환경 교체 로직 ---
            
            # 1. 교체 전, 모든 에이전트의 현재 상태와 누적 보상을 임시 저장합니다.
            print("모든 에이전트의 현재 상태와 누적 보상을 임시 저장합니다...")
            
            temp_state_files = []
            saved_rewards = env.env_method('get_cumulative_reward') # 이 함수를 위해 env에 get_cumulative_reward 추가 필요
            
            for i in range(num_cpu):
                path = (logs_path / f"temp_agent_{i}.state").resolve()
                env.env_method('save_current_state', str(path), indices=[i])
                temp_state_files.append(str(path))

            best_agent_state_path = temp_state_files[best_agent_idx]

            # 2. 기존의 모든 환경 프로세스를 완전히 종료합니다.
            print("기존 환경을 모두 종료합니다...")
            env.close()

            # 3. 새로운 환경에 주입할 초기 상태 경로 리스트를 준비합니다.
            half_point = num_cpu // 2
            init_state_paths = [None] * num_cpu
            target_group_is_first_half = best_agent_idx < half_point
            
            for i in range(num_cpu):
                # 최고 에이전트 그룹은 최고 상태로, 나머지는 각자의 이전 상태로 초기화
                if (target_group_is_first_half and i < half_point) or \
                   (not target_group_is_first_half and i >= half_point):
                    init_state_paths[i] = best_agent_state_path
                else:
                    init_state_paths[i] = temp_state_files[i]

            # 4. 새로운 환경 벡터를 생성합니다.
            print("새로운 환경을 생성합니다...")
            unwrapped_env = SubprocVecEnv([make_env(i, env_config, custom_init_state=init_state_paths[i]) for i in range(num_cpu)])
            env = VecTransposeImage(unwrapped_env)
            
            # 5. 새로 생성된 환경에 이전의 누적 보상을 다시 설정합니다.
            print("누적 보상을 복원합니다...")
            for i in range(num_cpu):
                env.env_method('set_cumulative_reward', saved_rewards[i], indices=[i])
            
            # 6. PPO 모델에 새로운 환경을 연결합니다.
            model.set_env(env)

            model._last_obs = env.reset()
            
            # 7. 사용했던 임시 상태 파일들을 모두 삭제합니다.
            for path_str in temp_state_files:
                p = Path(path_str)
                if p.exists():
                    p.unlink()

            print("환경 교체가 완료되었습니다.")
            

        # --- 1. 태스크 진행 여부 확인 및 목표 설정 ---
        all_game_states = env.env_method('get_structured_state')
        
        if task_manager.is_current_task_completed(all_game_states[best_agent_idx]):
            task_manager.advance_to_next_task()

        current_task_desc = task_manager.get_current_task_description()
        print(f"현재 주요 목표: {current_task_desc}")

        # --- 2. LLM을 통해 다음 스킬 결정 ---
        print("LLM 플래너를 호출하여 각 에이전트의 다음 목표 스킬을 결정합니다...")
        chosen_skills = llm_planner.choose_next_skill_batch(all_game_states, current_task_desc, AVAILABLE_SKILLS_RED)
        
        # --- 3. 각 환경에 목표 스킬 설정 ---
        for i, skill in enumerate(chosen_skills):
            env.env_method('set_current_skill', skill, indices=[i])

        # --- 4. 저수준 액터 학습 수행 ---
        print(f"저수준 PPO 액터가 {steps_per_segment} 스텝 동안 목표 스킬을 수행하며 학습합니다...")
        model.learn(
            total_timesteps=steps_per_segment, 
            reset_num_timesteps=False,
            callback=callbacks,
        )
        
        total_steps += steps_per_segment
        model.save(model_path / f"ppo_model_{total_steps}_steps")

    print("학습이 완료되었습니다.")
    print("최종 평가를 진행하여 최고 환경의 상태를 저장합니다...")
    # 1. 모든 환경의 최종 스탯을 가져옵니다.
    final_stats = env.env_method('get_agent_priority_stats')
    # 2. 가장 우수한 스탯과 해당 에이전트의 번호를 찾습니다.
    best_final_stats = max(final_stats)
    best_final_agent_idx = final_stats.index(best_final_stats)

    print(f"최고 성과 에이전트: {best_final_agent_idx}, 스탯: {best_final_stats}")

    # 3. 최고 에이전트의 .state 데이터를 가져옵니다.
    best_state_data = env.env_method('get_pyboy_state', indices=[best_final_agent_idx])[0]
    
    # 4. 해당 모델 저장 폴더에 'best_env_final.state' 파일로 저장합니다.
    save_path = model_path / "best_env_final.state"
    with open(save_path, "wb") as f:
        f.write(best_state_data)
    print(f"최고 환경의 상태를 '{save_path}'에 저장했습니다.")
    # --- 저장 로직 끝 ---

    model.save(model_path / "final_model.zip")
    env.close()