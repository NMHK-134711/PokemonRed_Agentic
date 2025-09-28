# llm_planner.py (클라이언트 버전)
import requests
from skill_library_red import AVAILABLE_SKILLS_RED # 스킬 객체 매핑을 위해 필요

class LLMPlanner:
    def __init__(self, server_url: str = "http://127.0.0.1:5000"):
        self.server_url = f"{server_url}/get_skills"
        print(f"LLM Planner가 서버({self.server_url})와 통신하도록 설정되었습니다.")

    def choose_next_skill_batch(self, game_states: list, agent_tasks: list, available_skills: list) -> list:
        """LLM 서버에 게임 상태와 목표를 보내고, 선택된 스킬 목록을 받아옵니다."""
        
        payload = {
            "game_states": game_states,
            "agent_tasks": agent_tasks
        }
        
        try:
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            chosen_skills_desc = response_data['chosen_skills']
            
            print(f"LLM 서버로부터 받은 스킬 설명: {chosen_skills_desc}")

            # 서버에서 받은 description을 실제 Skill 객체로 다시 매핑합니다.
            chosen_skills = [None] * len(game_states)
            skill_map = {skill.description: skill for skill in available_skills}

            for i, desc in enumerate(chosen_skills_desc):
                if desc in skill_map:
                    chosen_skills[i] = skill_map[desc]
                else:
                    print(f"경고: 서버가 반환한 스킬 '{desc}'를 로컬 스킬 라이브러리에서 찾을 수 없습니다. 기본 스킬을 할당합니다.")
                    chosen_skills[i] = available_skills[0]
            
            return chosen_skills

        except requests.exceptions.RequestException as e:
            print(f"LLM 서버 통신 오류: {e}")
            return [available_skills[0]] * len(game_states)