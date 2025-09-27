# llm_planner.py 
import re
from llama_cpp import Llama

class LLMPlanner:
    def __init__(self, model_path: str = "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"):
        print(f"'{model_path}' GGUF 모델을 로딩합니다. 시간이 걸릴 수 있습니다...")
        
        # llama-cpp-python v0.2.20을 사용하여 GGUF 모델 로드
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # GPU에 모든 레이어를 오프로드
            n_ctx=4096,       # 컨텍스트 크기 설정
            verbose=False     # llama.cpp의 자세한 로그는 끄기
        )
        print("모델 로딩이 완료되었습니다.")

    def _create_batch_prompt_string(self, game_states: list, main_task: str, available_skills: list) -> str:
        """(수정 없음) 여러 에이전트의 상태를 받아 하나의 전체 프롬프트 문자열을 생성합니다."""
        
        situation_reports = []
        for i, game_state in enumerate(game_states):
            loc = game_state['location']
            player = game_state['player_info']
            party_list = game_state['party_info']['pokemon']
            party_str = ", ".join([f"{p['species_name']}(Lv.{p['level']})" for p in party_list])
            report = (
                f"### Agent {i} State\n"
                f"- Location: {loc['map_name']} (X:{loc['x_coord']}, Y:{loc['y_coord']})\n"
                f"- Player: ${player['money']}, Badges: {player['kanto_badges_count']}\n"
                f"- Party: {party_str}"
            )
            situation_reports.append(report)

        all_situations = "\n\n".join(situation_reports)
        skill_descriptions = "\n".join([f"- {skill.description}" for skill in available_skills])

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert AI playing 'Pokémon Red'. For each agent, choose the single best action from the given list to achieve the current objective. Respond using the specified format for ALL agents.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Current Task
{main_task}

### Contextual Game Knowledge
- To challenge a Gym Leader, you must first be in the correct city.
- The Viridian City Gym is locked until the very end of the game. Do not attempt to enter it early.
- You must complete tasks in the order they are presented in the 'Current Task'.

### Current Game States
{all_situations}

### Available Actions
{skill_descriptions}

### Instructions
1. Analyze each agent's state in relation to the 'Current Task' and the 'Contextual Game Knowledge'.
2. Choose the single most logical and possible action from the 'Available Actions' list.
3. **Constraint:** Do not select an action that is currently impossible, such as entering a locked gym.
4. Provide your decision for every agent in the specified format, starting each on a new line.
Format:
Agent 0 Decision: [Copy the chosen action description here]
Agent 1 Decision: [Copy the chosen action description here]
...<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
    
    def choose_next_skill_batch(self, game_states: list, main_task: str, available_skills: list) -> list:
        """여러 에이전트의 다음 스킬을 한 번의 LLM 호출로 결정합니다."""
        prompt_string = self._create_batch_prompt_string(game_states, main_task, available_skills)
        
        # llama-cpp-python v0.2.20 모델을 사용하여 텍스트 생성
        # 중요: v0.2.20에서는 'prompt=' 인자 이름 없이 문자열을 직접 전달해야 합니다.
        output = self.model(
            prompt=prompt_string,  # <--- 'prompt='를 다시 추가!
            max_tokens=512,
            temperature=0.1,
            stop=["<|eot_id|>"]
        )
        
        response_text = output['choices'][0]['text']
        
        print(f"LLM 원본 응답 (배치):\n{response_text}")

        # 파싱 로직 (수정 없음)
        chosen_skills = [None] * len(game_states)
        decisions = re.findall(r"Agent (\d+) Decision: (.*)", response_text)

        for agent_idx_str, desc in decisions:
            agent_idx = int(agent_idx_str)
            if agent_idx < len(game_states):
                best_match_skill = None
                for skill in available_skills:
                    if skill.description in desc.strip():
                        best_match_skill = skill
                        break
                if best_match_skill:
                    chosen_skills[agent_idx] = best_match_skill
        
        for i in range(len(chosen_skills)):
            if chosen_skills[i] is None:
                print(f"경고: LLM이 Agent {i}의 스킬을 선택하지 못했습니다. 기본 스킬을 할당합니다.")
                # 'chosen_songs' -> 'chosen_skills' 오타 수정
                chosen_skills[i] = available_skills[0]
        
        return chosen_skills