import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

class LLMPlanner:
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print(f"'{model_id}' 모델을 로딩합니다. 시간이 걸릴 수 있습니다...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            #torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("모델 로딩이 완료되었습니다.")

    def _create_prompt_messages(self, game_state: dict, main_task: str, available_skills: list) -> list:
        loc = game_state['location']
        player = game_state['player_info']
        party_list = game_state['party_info']['pokemon']
        party_str = ", ".join([f"Lv.{p['level']}" for p in party_list])

        current_situation = (
            f"현재 위치: 맵(Bank {loc['map_bank']}, ID {loc['map_id']}).\n"
            f"플레이어 정보: 돈 ${player['money']}, 배지 {player['johto_badges_count']}개.\n"
            f"파티 정보: {game_state['party_info']['count']}마리 ({party_str}).\n"
            f"주요 이벤트: {game_state['event_statuses']}"
        )
        skill_descriptions = "\n".join([f"- {skill.description}" for skill in available_skills])

        system_prompt = "You are an expert AI playing 'Pokémon Gold'. Your task is to choose the single best action from a given list to achieve the main objective. Respond using the specified format."
        user_prompt = (
            f"### Current Task\n{main_task}\n\n"
            f"### Contextual Game Knowledge\n"
            f"- To challenge a Gym Leader, you must first be in the correct city.\n"
            f"- The Viridian City Gym is locked until the very end of the game. Do not attempt to enter it early.\n"
            f"- You must complete tasks in the order they are presented in the 'Current Task'.\n\n"
            f"### Current Game States\n{current_situation}\n\n"
            f"### Available Actions\n{skill_descriptions}\n\n"
            "### Instructions\n"
            "1. Analyze each agent's state in relation to the 'Current Task' and the 'Contextual Game Knowledge'.\n"
            "2. Choose the single most logical and possible action from the 'Available Actions' list.\n"
            "3. **Constraint:** Do not select an action that is currently impossible, such as entering a locked gym.\n"
            "4. Provide your decision for every agent in the specified format, starting each on a new line.\n"
            "Format:\n"
            "Agent 0 Decision: [Copy the chosen action description here]\n"
            "Agent 1 Decision: [Copy the chosen action description here]\n"
            "..."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def choose_next_skill(self, game_state: dict, main_task: str, available_skills: list):
        messages = self._create_prompt_messages(game_state, main_task, available_skills)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 1. [핵심] `apply_chat_template`이 딕셔너리가 아닌 'Tensor'를 반환하는 것을 전제로 합니다.
        #    따라서 `inputs` 변수를 `inputs_tensor`로 명명하여 텐서임을 명확히 합니다.
        inputs_tensor = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device) # 텐서이므로 바로 .to(device)를 호출할 수 있습니다.
        
        # 2. [수정] `inputs_tensor`를 `input_ids` 인자에 직접 전달합니다.
        #    이 방식으로는 attention_mask를 전달할 수 없어 경고가 발생할 수 있지만, 실행은 됩니다.
        outputs = self.model.generate(
            input_ids=inputs_tensor,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 3. [수정] 프롬프트의 길이는 `inputs_tensor`의 shape에서 직접 가져옵니다.
        #    shape[0]은 배치 크기(1), shape[1]은 시퀀스 길이입니다.
        prompt_length = inputs_tensor.shape[1]
        response_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        print(f"LLM 원본 응답:\n{response_text}")

        try:
            chosen_description = response_text.split('Decision:')[1].strip()
        except IndexError:
            chosen_description = ""
        
        best_match_skill = None
        for skill in available_skills:
            if skill.description in chosen_description:
                best_match_skill = skill
                break
        
        if best_match_skill:
            print(f"LLM 선택 (파싱): {best_match_skill.description}")
            return best_match_skill
        else:
            print("경고: LLM이 유효한 스킬을 선택하지 못했습니다. 기본 스킬을 반환합니다.")
            return available_skills[0]
        
    def _create_batch_prompt_messages(self, game_states: list, main_task: str, available_skills: list) -> list:
        """여러 에이전트의 상태를 받아 하나의 배치 프롬프트를 생성합니다."""
        
        situation_reports = []
        for i, game_state in enumerate(game_states):
            # [수정] 레드 버전에 맞는 정보로 프롬프트 구성
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

        # [수정] 시스템 프롬프트와 유저 프롬프트 내용 개선
        system_prompt = "You are an expert AI playing 'Pokémon Red'. For each agent, choose the single best action from the given list to achieve the current objective. Respond using the specified format for ALL agents."
        user_prompt = (
            # main_task는 이제 "사가"와 "목표"를 모두 포함
            f"### Current Task\n{main_task}\n\n"
            f"### Current Game States\n{all_situations}\n\n"
            f"### Available Actions\n{skill_descriptions}\n\n"
            "### Instructions\n"
            "1. Analyze each agent's state in relation to the 'Current Task'.\n"
            "2. For each agent, choose the single most logical action from the 'Available Actions' list to make progress.\n"
            "3. Provide your decision for every agent in the specified format, starting each on a new line.\n"
            "Format:\n"
            "Agent 0 Decision: [Copy the chosen action description here]\n"
            "Agent 1 Decision: [Copy the chosen action description here]\n"
            "..."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
    def choose_next_skill_batch(self, game_states: list, main_task: str, available_skills: list) -> list:
        """여러 에이전트의 다음 스킬을 한 번의 LLM 호출로 결정합니다."""
        messages = self._create_batch_prompt_messages(game_states, main_task, available_skills)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs_tensor = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # 더 긴 응답을 위해 max_new_tokens를 늘려줍니다.
        outputs = self.model.generate(
            input_ids=inputs_tensor,
            max_new_tokens=512, # 에이전트 수에 비례하여 늘려야 할 수 있음
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        prompt_length = inputs_tensor.shape[1]
        response_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        print(f"LLM 원본 응답 (배치):\n{response_text}")

        # 파싱 로직
        chosen_skills = [None] * len(game_states)
        # "Agent X Decision:" 패턴으로 각 줄을 찾습니다.
        decisions = re.findall(r"Agent (\d+) Decision: (.*)", response_text)

        for agent_idx_str, desc in decisions:
            agent_idx = int(agent_idx_str)
            if agent_idx < len(game_states):
                # 가장 잘 맞는 스킬을 찾습니다.
                best_match_skill = None
                for skill in available_skills:
                    if skill.description in desc.strip():
                        best_match_skill = skill
                        break
                if best_match_skill:
                    chosen_skills[agent_idx] = best_match_skill
        
        # LLM이 선택하지 못한 에이전트는 기본 스킬(예: 첫 번째 스킬)로 대체합니다.
        for i in range(len(chosen_skills)):
            if chosen_skills[i] is None:
                print(f"경고: LLM이 Agent {i}의 스킬을 선택하지 못했습니다. 기본 스킬을 할당합니다.")
                chosen_skills[i] = available_skills[0]
        
        return chosen_skills