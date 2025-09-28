# llm_server.py (독립 실행형으로 수정 완료)
from flask import Flask, request, jsonify
from llama_cpp import Llama
from skill_library_red import AVAILABLE_SKILLS_RED
import re

# --- LLM 관련 로직을 모두 서버 파일 안으로 이동 ---

def _create_batch_prompt_string(game_states: list, agent_tasks: list, available_skills: list) -> str:
    """각 에이전트의 개별 목표를 명확히 하고, LLM이 규칙을 따르도록 프롬프트를 생성합니다."""
    situation_reports = []
    for i, game_state in enumerate(game_states):
        loc = game_state.get('location', {})
        player = game_state.get('player_info', {})
        party_list = game_state.get('party_info', {}).get('pokemon', [])
        party_str = ", ".join([f"{p.get('species_name', 'N/A')}(Lv.{p.get('level', 'N/A')})" for p in party_list])
        
        agent_task_desc = "No task assigned"
        if i < len(agent_tasks) and agent_tasks[i] and len(agent_tasks[i].splitlines()) > 1:
            agent_task_desc = agent_tasks[i].splitlines()[1].replace("구체적인 목표: ", "").strip()
        
        report = (
            f"### Agent {i} Status\n"
            f"- **Individual Objective**: {agent_task_desc}\n"
            f"- Location: {loc.get('map_name', 'N/A')} (X:{loc.get('x_coord', 'N/A')}, Y:{loc.get('y_coord', 'N/A')})\n"
            f"- Player: ${player.get('money', 'N/A')}, Badges: {player.get('kanto_badges_count', 'N/A')}\n"
            f"- Party: {party_str}"
        )
        situation_reports.append(report)

    all_situations = "\n\n".join(situation_reports)
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
...<|e_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

def choose_skills_from_llm(llm_model, game_states: list, agent_tasks: list, available_skills: list) -> list[str]:
    """주어진 정보를 바탕으로 LLM을 호출하고, 스킬 설명 문자열 리스트를 반환합니다."""
    prompt_string = _create_batch_prompt_string(game_states, agent_tasks, available_skills)
    
    output = llm_model(
        prompt=prompt_string,
        max_tokens=512,
        temperature=0.0,
        stop=["<|eot_id|>"]
    )
    
    response_text = output['choices'][0]['text']
    print(f"LLM 원본 응답 (배치):\n{response_text}")

    # description 문자열만 파싱하여 리스트로 만듭니다.
    chosen_skill_descs = [""] * len(game_states)
    decisions = re.findall(r"Agent\s*(\d+)\s*Decision:\s*(.*)", response_text)

    for agent_idx_str, desc in decisions:
        agent_idx = int(agent_idx_str)
        if agent_idx < len(game_states):
            chosen_skill_descs[agent_idx] = desc.strip()
            
    # LLM이 응답을 누락한 경우, 안전하게 첫 번째 스킬 설명으로 채웁니다.
    skill_map = {skill.description for skill in available_skills}
    for i in range(len(chosen_skill_descs)):
        if chosen_skill_descs[i] not in skill_map:
            print(f"경고: LLM 응답이 유효하지 않거나 누락되어 Agent {i}의 스킬을 기본값으로 설정합니다.")
            chosen_skill_descs[i] = available_skills[0].description

    return chosen_skill_descs

# --- Flask 서버 설정 ---

app = Flask(__name__)

# 서버가 시작될 때 LLM 모델을 한 번만 로드합니다.
print("LLM 서버 초기화 중...")
llm_model = Llama(
    model_path="Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)
print("LLM 서버 준비 완료. 요청을 기다립니다...")

@app.route('/get_skills', methods=['POST'])
def get_skills_endpoint():
    try:
        data = request.json
        game_states = data['game_states']
        agent_tasks = data['agent_tasks']
        
        # 서버 내부에 포함된 로직을 호출합니다.
        chosen_skills_desc = choose_skills_from_llm(
            llm_model,
            game_states,
            agent_tasks,
            AVAILABLE_SKILLS_RED
        )
        
        return jsonify({"chosen_skills": chosen_skills_desc})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)