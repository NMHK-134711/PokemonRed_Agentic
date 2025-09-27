# llm_planner.py (프롬프트 강화 완료)
import re
from llama_cpp import Llama

class LLMPlanner:
    def __init__(self, model_path: str = "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"):
        print(f"'{model_path}' GGUF 모델을 로딩합니다. 시간이 걸릴 수 있습니다...")
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False
        )
        print("모델 로딩이 완료되었습니다.")

    def _create_batch_prompt_string(self, game_states: list, agent_tasks: list, available_skills: list) -> str:
        """
        [수정됨] 각 에이전트의 개별 목표를 명확히 하고, LLM이 규칙을 따르도록 프롬프트를 대폭 강화합니다.
        """
        situation_reports = []
        for i, game_state in enumerate(game_states):
            loc = game_state['location']
            player = game_state['player_info']
            party_list = game_state['party_info']['pokemon']
            party_str = ", ".join([f"{p['species_name']}(Lv.{p['level']})" for p in party_list])
            
            # 각 에이전트의 개별 목표를 가져옵니다.
            # "현재 사가: ..." 부분을 제거하고 구체적인 목표만 남깁니다.
            agent_task_desc = agent_tasks[i].splitlines()[1].replace("구체적인 목표: ", "").strip()
            
            report = (
                f"### Agent {i} Status\n"
                f"- **Individual Objective**: {agent_task_desc}\n"
                f"- Location: {loc['map_name']} (X:{loc['x_coord']}, Y:{loc['y_coord']})\n"
                f"- Player: ${player['money']}, Badges: {player['kanto_badges_count']}\n"
                f"- Party: {party_str}"
            )
            situation_reports.append(report)

        all_situations = "\n\n".join(situation_reports)
        # 스킬 목록을 명확하게 번호 매기기하여 LLM이 선택하기 쉽게 만듭니다.
        skill_descriptions = "\n".join([f"{i+1}. {skill.description}" for i, skill in enumerate(available_skills)])

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise and methodical AI agent controller for the game 'Pokémon Red'. Your ONLY function is to select the most appropriate action from a predefined list for multiple agents based on their state and objectives. You MUST follow all formatting rules precisely. Do not add any extra conversation or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Agent Status & Individual Objectives
{all_situations}

### Action Library (You MUST choose from this list)
{skill_descriptions}

### YOUR TASK
1. For each agent, analyze its status in relation to its unique **'Individual Objective'**.
2. Your response MUST select one action for each agent from the **'Action Library'**.
3. The selected action's text **MUST be an EXACT, character-for-character copy** from the 'Action Library' (excluding the number).
4. **DO NOT** use the text from the agent's objective. **DO NOT** add any extra words.
5. Provide a decision for ALL agents.

### RESPONSE FORMAT
Your response MUST strictly follow this format, with each agent on a new line:
Agent 0 Decision: [Exact action description copied from the Action Library]
Agent 1 Decision: [Exact action description copied from the Action Library]
...<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
    
    def choose_next_skill_batch(self, game_states: list, agent_tasks: list, available_skills: list) -> list:
        # 이 함수의 시그니처가 프롬프트 함수와 일치하도록 수정
        prompt_string = self._create_batch_prompt_string(game_states, agent_tasks, available_skills)
        
        output = self.model(
            prompt=prompt_string,
            max_tokens=512,
            temperature=0.0, # 창의성을 없애기 위해 온도를 0으로 설정
            stop=["<|eot_id|>"]
        )
        
        response_text = output['choices'][0]['text']
        print(f"LLM 원본 응답 (배치):\n{response_text}")

        chosen_skills = [None] * len(game_states)
        # 정규표현식이 더 많은 공백이나 변형에 대응할 수 있도록 수정
        decisions = re.findall(r"Agent\s*(\d+)\s*Decision:\s*(.*)", response_text)

        for agent_idx_str, desc in decisions:
            agent_idx = int(agent_idx_str)
            desc_stripped = desc.strip()
            if agent_idx < len(game_states):
                best_match_skill = None
                # 가장 일치하는 스킬을 찾습니다.
                for skill in available_skills:
                    if skill.description == desc_stripped:
                        best_match_skill = skill
                        break
                if best_match_skill:
                    chosen_skills[agent_idx] = best_match_skill
        
        for i in range(len(chosen_skills)):
            if chosen_skills[i] is None:
                print(f"경고: LLM이 Agent {i}의 스킬을 선택하지 못했습니다. 기본 스킬을 할당합니다.")
                # 실패 시 기본 스킬로 GoToMapSkillRed("VIRIDIAN CITY") 같은 안전한 스킬을 지정하는 것이 좋습니다.
                # 여기서는 일단 첫 번째 스킬을 유지합니다.
                chosen_skills[i] = available_skills[0]
        
        return chosen_skills