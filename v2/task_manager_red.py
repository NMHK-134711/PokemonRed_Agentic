# task_manager_red.py
import json

class TaskManagerRed:
    def __init__(self, plan_path: str):
        with open(plan_path, 'r', encoding='utf-8') as f:
            self.plan = json.load(f)
        
        self.current_saga_index = 0
        self.current_objective_index = 0
        self.game_state = {}

    def _update_internal_state(self, game_state: dict):
        self.game_state = game_state

    # --- 검증 헬퍼 함수 ---
    def _has_item(self, item_name: str) -> bool:
        inventory = self.game_state.get('inventory', {}).get('items', [])
        return any(item[0].upper() == item_name.upper() for item in inventory)

    def _check_badges(self, count: int) -> bool:
        return self.game_state.get('player_info', {}).get('kanto_badges_count', 0) >= count

    def _is_in_location(self, map_name: str) -> bool:
        return self.game_state.get('location', {}).get('map_name') == map_name.upper()
    
    # [신규] Elite Four 격파 여부 확인을 위한 헬퍼 함수
    # 참고: 이 기능이 제대로 동작하려면 game_state_red.py가 Elite Four 격파 관련
    # 메모리 주소(예: 0xD746 ~)를 읽어 event_statuses에 추가해야 합니다.
    def _check_event_flag(self, event_name: str) -> bool:
        return self.game_state.get('event_statuses', {}).get(event_name, False)

    def get_current_task_description(self) -> str:
        """현재 수행해야 할 구체적인 목표(Objective)와 전체 사가(Saga)를 함께 반환합니다."""
        if self.current_saga_index >= len(self.plan['tasks']):
            return "모든 목표를 달성했습니다!"

        current_saga = self.plan['tasks'][self.current_saga_index]
        if self.current_objective_index >= len(current_saga['objectives']):
            return f"사가 '{current_saga['saga']}'의 모든 목표를 달성했습니다! 다음 사가로 진행합니다."
        
        saga_title = current_saga['saga']
        objective_desc = current_saga['objectives'][self.current_objective_index]
        
        return f"현재 사가: {saga_title}\n구체적인 목표: {objective_desc}"

    def is_current_task_completed(self, game_state: dict) -> bool:
        """plan.json의 각 목표(Objective)를 게임 상태와 대조하여 완료 여부를 확인합니다."""
        self._update_internal_state(game_state)
        
        if self.current_saga_index >= len(self.plan['tasks']): return True
        current_saga = self.plan['tasks'][self.current_saga_index]
        if self.current_objective_index >= len(current_saga['objectives']): return True
        
        objective = current_saga['objectives'][self.current_objective_index]

        # --- 각 Objective에 대한 상세한 완료 조건 ---
        # Saga 1
        if "buy Poké Balls" in objective: return self._has_item("POKé BALL")
        if "rival battle" in objective and "Route 22" in objective: return self._is_in_location("VIRIDIAN FOREST")
        if "enter Viridian Forest" in objective: return self._is_in_location("VIRIDIAN FOREST")
        if "catching a Caterpie and a Pidgey" in objective: return self._is_in_location("PEWTER CITY")
        if "arrive in Pewter City" in objective: return self._is_in_location("PEWTER CITY")
        if "defeat Gym Leader Brock" in objective: return self._check_badges(1)
            
        # Saga 2
        if "enter Mt. Moon" in objective: return self._is_in_location("MT. MOON 1F")
        if "obtain one of the two fossils" in objective: return self._has_item("DOME FOSSIL") or self._has_item("HELIX FOSSIL")
        if "reach Cerulean City" in objective: return self._is_in_location("CERULEAN CITY")
        if "Nugget Bridge" in objective: return self._is_in_location("ROUTE 25")
        if "receive the S.S. Ticket" in objective: return self._has_item("S.S. TICKET")
        if "defeat Gym Leader Misty" in objective: return self._check_badges(2)
        
        # Saga 3
        if "battle a Team Rocket grunt" in objective: return self._is_in_location("ROUTE 5")
        if "enter the Underground Path" in objective: return self._is_in_location("VERMILION CITY")
        if "Arrive in Vermilion City" in objective: return self._is_in_location("VERMILION CITY")
        if "Board the S.S. Anne" in objective: return self._has_item("HM01") # 배에서 내리면 배가 떠나므로 Cut 획득을 프록시로 사용
        if "obtain HM01 (Cut)" in objective: return self._has_item("HM01")
        if "Teach Cut" in objective: return self._check_badges(3) # 행동이므로 다음 목표 달성을 프록시로 사용
        if "defeat Gym Leader Lt. Surge" in objective: return self._check_badges(3)

        # Saga 4
        if "enter Diglett's Cave" in objective: return self._is_in_location("DIGLETTS CAVE")
        if "reach the east side of Route 2" in objective: return self._has_item("HM05") # 플래시 획득을 프록시로 사용
        if "obtain HM05 (Flash)" in objective: return self._has_item("HM05")
        if "access Route 9" in objective: return self._is_in_location("ROCK TUNNEL 1F")
        if "enter Rock Tunnel" in objective: return self._is_in_location("ROCK TUNNEL 1F")
        if "arrive in Lavender Town" in objective: return self._is_in_location("LAVENDER TOWN")
        if "Underground Path to Route 7" in objective: return self._is_in_location("CELADON CITY")
        if "Arrive in Celadon City" in objective: return self._is_in_location("CELADON CITY")
        if "defeat Gym Leader Erika" in objective: return self._check_badges(4)

        # Saga 5
        if "find the secret hideout" in objective: return self._has_item("SILPH SCOPE") # 로켓단 아지트 클리어 보상
        if "Navigate the Team Rocket Hideout" in objective: return self._has_item("SILPH SCOPE")
        if "Obtain the Silph Scope" in objective: return self._has_item("SILPH SCOPE")
        if "enter the Pokémon Tower" in objective: return self._is_in_location("POKEMON TOWER 1F")
        if "rescue Mr. Fuji" in objective: return self._has_item("POKé FLUTE")
        if "Receive the Poké Flute" in objective: return self._has_item("POKé FLUTE")
        if "wake up and defeat/capture Snorlax" in objective: return self._is_in_location("FUCHSIA CITY")
        if "reach Fuchsia City" in objective: return self._is_in_location("FUCHSIA CITY")
        if "defeat Gym Leader Koga" in objective: return self._check_badges(5)

        # Saga 6
        if "Enter the Safari Zone" in objective: return self._is_in_location("SAFARI ZONE ENTRANCE")
        if "obtain HM03 (Surf)" in objective: return self._has_item("HM03")
        if "receive HM04 (Strength)" in objective: return self._has_item("HM04")
        if "Teach Surf and Strength" in objective: return self._is_in_location("SAFFRON CITY")
        if "travel to Saffron City" in objective: return self._is_in_location("SAFFRON CITY")
        if "Navigate the 11 floors of Silph Co." in objective: return self._has_item("MASTER BALL") # 실프주식회사 클리어 보상
        if "Receive a Master Ball" in objective: return self._has_item("MASTER BALL")
        if "defeat Gym Leader Sabrina" in objective: return self._check_badges(6)

        # Saga 7
        if "reach Cinnabar Island" in objective: return self._is_in_location("CINNABAR ISLAND")
        if "find the Secret Key" in objective: return self._has_item("SECRET KEY")
        if "Defeat Gym Leader Blaine" in objective: return self._check_badges(7)
        
        # Saga 8
        if "challenge the final gym" in objective: return self._check_badges(8)
        if "Defeat Gym Leader Giovanni" in objective: return self._check_badges(8)

        # Saga 9
        if "reach Victory Road" in objective: return self._is_in_location("VICTORY ROAD 2F") # 1,2,3층 중 하나
        if "reach the Indigo Plateau" in objective: return self._is_in_location("INDIGO PLATEAU LOBBY")
        if "Defeat Elite Four Lorelei" in objective: return self._check_event_flag('defeated_lorelei')
        if "Defeat Elite Four Bruno" in objective: return self._check_event_flag('defeated_bruno')
        if "Defeat Elite Four Agatha" in objective: return self._check_event_flag('defeated_agatha')
        if "Defeat Elite Four Lance" in objective: return self._check_event_flag('defeated_lance')
        if "defeat the Champion" in objective: return self._check_event_flag('became_champion')

        return False

    def advance_to_next_task(self):
        """다음 목표(Objective)로 넘어갑니다. 사가가 끝나면 다음 사가로 이동합니다."""
        if self.current_saga_index >= len(self.plan['tasks']): return

        current_saga = self.plan['tasks'][self.current_saga_index]
        self.current_objective_index += 1
        
        if self.current_objective_index >= len(current_saga['objectives']):
            self.current_saga_index += 1
            self.current_objective_index = 0
            if self.current_saga_index < len(self.plan['tasks']):
                print(f"\n***** 다음 사가로 진행합니다: {self.plan['tasks'][self.current_saga_index]['saga']} *****\n")

        print(f"***** 다음 목표로 진행합니다: {self.get_current_task_description()} *****")

    def sync_with_initial_state(self, initial_game_state: dict):
        print("저장된 게임 상태와 목표 리스트를 동기화합니다...")
        while self.is_current_task_completed(initial_game_state):
             if self.current_saga_index >= len(self.plan['tasks']): break
             print(f"이미 완료된 목표: '{self.get_current_task_description()}' -> 건너뜁니다.")
             self.advance_to_next_task()
        print(f"동기화 완료! 현재 목표: '{self.get_current_task_description()}'")