# skill_library_red.py
# =================================
# 레드 버전용 스킬 클래스 정의
# =================================

class Skill:
    """The abstract base class for all skills."""
    def __init__(self, description: str):
        self.description = description

    def is_achieved(self, prev_state: dict, current_state: dict) -> bool:
        """Checks if the objective of this skill has been achieved."""
        raise NotImplementedError

    def get_reward(self, prev_state: dict, current_state: dict) -> float:
        """Provides a large, immediate reward upon skill completion."""
        if self.is_achieved(prev_state, current_state) and not self.is_achieved({}, prev_state):
            return 500.0  # Achievement reward
        return 0.0

class GoToMapSkillRed(Skill):
    def __init__(self, map_name: str):
        super().__init__(f"Go to {map_name}")
        self.map_name = map_name.upper()
    def is_achieved(self, _, current_state) -> bool:
        return current_state.get('location', {}).get('map_name') == self.map_name

class DefeatGymLeaderSkillRed(Skill):
    def __init__(self, badge_name: str, target_badge_count: int):
        super().__init__(f"Defeat the Gym Leader to get the {badge_name} Badge")
        self.target_badge_count = target_badge_count
    def is_achieved(self, _, current_state) -> bool:
        return current_state.get('player_info', {}).get('kanto_badges_count', 0) >= self.target_badge_count

class ObtainItemSkillRed(Skill):
    def __init__(self, item_name: str):
        super().__init__(f"Obtain the item: {item_name}")
        self.item_name = item_name.upper()
    def is_achieved(self, _, current_state) -> bool:
        inventory = current_state.get('inventory', {}).get('items', [])
        return any(item[0].upper() == self.item_name for item in inventory)

class TeachHMSkill(Skill):
    def __init__(self, hm_name: str):
        super().__init__(f"Teach a Pokémon {hm_name}")
    def is_achieved(self, _, current_state) -> bool:
        return True # 행동 자체를 목표로 하므로, 선택되면 달성된 것으로 간주

class CapturePokemonSkill(Skill):
    def __init__(self, species_name: str):
        super().__init__(f"Capture a wild {species_name}")
        self.species_name = species_name.upper()
    def is_achieved(self, _, current_state) -> bool:
        party = current_state.get('party_info', {}).get('pokemon', [])
        return any(p['species_name'] == self.species_name for p in party)

class DefeatEliteFourSkill(Skill):
    def __init__(self, member_name: str):
        super().__init__(f"Defeat Elite Four member: {member_name}")
        self.event_flag = f"defeated_{member_name.lower()}"
    def is_achieved(self, _, current_state) -> bool:
        return current_state.get('event_statuses', {}).get(self.event_flag, False)

class LevelUpSkill(Skill):
    """Skill for leveling up a party Pokémon."""
    def __init__(self, target_level: int):
        super().__init__(f"Level up a party Pokémon to level {target_level} or higher")
        self.target_level = target_level

    def is_achieved(self, prev_state, current_state) -> bool:
        party = current_state['party_info']['pokemon']
        if not party: return False
        return any(p['level'] >= self.target_level for p in party)

# =================================
# plan.json의 Saga에 맞춰 스킬을 그룹화
# =================================

SAGA_1_SKILLS = [
    GoToMapSkillRed(map_name="VIRIDIAN CITY"), 
    ObtainItemSkillRed(item_name="POKé BALL"),
    GoToMapSkillRed(map_name="ROUTE 22"), 
    GoToMapSkillRed(map_name="VIRIDIAN FOREST"),
    GoToMapSkillRed(map_name="PEWTER CITY"), 
    DefeatGymLeaderSkillRed(badge_name="Boulder", target_badge_count=1),
]
SAGA_2_SKILLS = [
    GoToMapSkillRed(map_name="ROUTE 3"), 
    GoToMapSkillRed(map_name="MT. MOON 1F"),
    ObtainItemSkillRed(item_name="DOME FOSSIL"), 
    GoToMapSkillRed(map_name="ROUTE 4"),
    GoToMapSkillRed(map_name="CERULEAN CITY"), 
    GoToMapSkillRed(map_name="ROUTE 24"),
    GoToMapSkillRed(map_name="ROUTE 25"), 
    ObtainItemSkillRed(item_name="S.S. TICKET"),
    DefeatGymLeaderSkillRed(badge_name="Cascade", target_badge_count=2),
]
SAGA_3_SKILLS = [
    GoToMapSkillRed(map_name="ROUTE 5"), 
    GoToMapSkillRed(map_name="VERMILION CITY"),
    GoToMapSkillRed(map_name="S.S. ANNE 1F"), 
    ObtainItemSkillRed(item_name="HM01"),
    TeachHMSkill(hm_name="Cut"), 
    DefeatGymLeaderSkillRed(badge_name="Thunder", target_badge_count=3),
]
SAGA_4_SKILLS = [
    GoToMapSkillRed(map_name="DIGLETTS CAVE"), 
    GoToMapSkillRed(map_name="ROUTE 2"),
    ObtainItemSkillRed(item_name="HM05"), 
    GoToMapSkillRed(map_name="ROUTE 9"),
    GoToMapSkillRed(map_name="ROCK TUNNEL 1F"), 
    GoToMapSkillRed(map_name="LAVENDER TOWN"),
    GoToMapSkillRed(map_name="CELADON CITY"), 
    DefeatGymLeaderSkillRed(badge_name="Rainbow", target_badge_count=4),
]
SAGA_5_SKILLS = [
    GoToMapSkillRed(map_name="GAME CORNER"), 
    ObtainItemSkillRed(item_name="SILPH SCOPE"),
    GoToMapSkillRed(map_name="POKEMON TOWER 1F"), 
    ObtainItemSkillRed(item_name="POKé FLUTE"),
    GoToMapSkillRed(map_name="ROUTE 12"), 
    GoToMapSkillRed(map_name="FUCHSIA CITY"),
    DefeatGymLeaderSkillRed(badge_name="Soul", target_badge_count=5),
]
SAGA_6_SKILLS = [
    GoToMapSkillRed(map_name="SAFARI ZONE ENTRANCE"), 
    ObtainItemSkillRed(item_name="HM03"),
    ObtainItemSkillRed(item_name="HM04"), 
    TeachHMSkill(hm_name="Surf"), 
    TeachHMSkill(hm_name="Strength"),
    GoToMapSkillRed(map_name="SAFFRON CITY"), 
    GoToMapSkillRed(map_name="SILPH CO. 1F"),
    ObtainItemSkillRed(item_name="MASTER BALL"), 
    DefeatGymLeaderSkillRed(badge_name="Marsh", target_badge_count=6),
]
SAGA_7_SKILLS = [
    GoToMapSkillRed(map_name="CINNABAR ISLAND"), 
    GoToMapSkillRed(map_name="POKEMON MANSION 1F"),
    ObtainItemSkillRed(item_name="SECRET KEY"), 
    DefeatGymLeaderSkillRed(badge_name="Volcano", target_badge_count=7),
]
SAGA_8_SKILLS = [
    GoToMapSkillRed(map_name="VIRIDIAN GYM"), 
    DefeatGymLeaderSkillRed(badge_name="Earth", target_badge_count=8),
]
SAGA_9_SKILLS = [
    GoToMapSkillRed(map_name="ROUTE 23"), 
    GoToMapSkillRed(map_name="VICTORY ROAD 2F"),
    GoToMapSkillRed(map_name="INDIGO PLATEAU LOBBY"), 
    DefeatEliteFourSkill(member_name="Lorelei"),
    DefeatEliteFourSkill(member_name="Bruno"), 
    DefeatEliteFourSkill(member_name="Agatha"),
    DefeatEliteFourSkill(member_name="Lance"), 
    DefeatEliteFourSkill(member_name="Champion"),
]
GENERAL_SKILLS = [
    LevelUpSkill(target_level=10), LevelUpSkill(target_level=15),
    LevelUpSkill(target_level=20), LevelUpSkill(target_level=25),
    LevelUpSkill(target_level=30), LevelUpSkill(target_level=35),
    LevelUpSkill(target_level=40), LevelUpSkill(target_level=45),
    LevelUpSkill(target_level=50), LevelUpSkill(target_level=55),
]

# 최종 스킬 리스트는 모든 스킬을 통합하여 중복 제거
unique_skills_list = set(
    SAGA_1_SKILLS + SAGA_2_SKILLS + SAGA_3_SKILLS + SAGA_4_SKILLS + 
    SAGA_5_SKILLS + SAGA_6_SKILLS + SAGA_7_SKILLS + SAGA_8_SKILLS + 
    SAGA_9_SKILLS + GENERAL_SKILLS 
)

unique_skills_dict = {skill.description: skill for skill in unique_skills_list}
AVAILABLE_SKILLS_RED = sorted(list(unique_skills_dict.values()), key=lambda x: x.description)