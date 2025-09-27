import json
from memory_reader import PokemonRedReader

# events.json을 미리 로드하여 재사용합니다.
with open("events.json", "r") as f:
    EVENT_NAME_MAP = json.load(f)

def read_all_event_flags(pyboy) -> dict:
    """
    events.json을 참조하여 게임 내 모든 이벤트 플래그의 상태를 읽고,
    알기 쉬운 이름과 함께 boolean 값으로 매핑된 딕셔너리를 반환합니다.
    """
    event_flags = {}
    # Elite Four와 Champion 플래그 이름 매핑 (task_manager에서 사용하는 이름 기준)
    e4_champion_map = {
        "Beat Loreleis Room Trainer 0": "defeated_lorelei",
        "Beat Brunos Room Trainer 0": "defeated_bruno",
        "Beat Agathas Room Trainer 0": "defeated_agatha",
        "Beat Lance": "defeated_lance",
        "Hall Of Fame Dex Rating": "became_champion"
    }

    # events.json에 정의된 모든 주소와 비트를 순회
    for key, name in EVENT_NAME_MAP.items():
        try:
            address_str, bit_str = key.split('-')
            address = int(address_str, 16)
            bit = int(bit_str)
            
            # 메모리에서 값 읽기
            value = pyboy.memory[address]
            is_set = (value & (1 << bit)) != 0
            
            # task_manager에서 사용할 이름으로 변환
            if name in e4_champion_map:
                event_key = e4_champion_map[name]
                event_flags[event_key] = is_set
            # 필요에 따라 다른 이벤트들도 추가할 수 있습니다.
            # 예: event_flags[name.lower().replace(" ", "_")] = is_set

        except (ValueError, IndexError, KeyError) as e:
            print(f"'{key}': '{name}' 키를 파싱하는 중 오류 발생 - {e}")
            continue
            
    return event_flags

def get_game_state(pyboy) -> dict:
    """
    PokemonRedReader를 사용하여 주요 게임 정보를 추출하고,
    Agentic AI 시스템이 요구하는 형식의 dictionary로 변환하여 반환합니다.
    """
    reader = PokemonRedReader(pyboy.memory)

    # --- 1. 위치 정보 ---
    x_coord, y_coord = reader.read_coordinates()
    map_id = pyboy.memory[0xD35E]
    location = {
        "map_id": map_id,
        "map_name": reader.read_location(),
        "x_coord": x_coord,
        "y_coord": y_coord,
    }

    # --- 2. 플레이어 정보 ---
    player_info = {
        "money": reader.read_money(),
        "kanto_badges_count": len(reader.read_badges()),
    }

    # --- 3. 파티 정보 ---
    party_data = reader.read_party_pokemon()
    party_pokemon = [
        {"species_name": p.species_name, "level": p.level} for p in party_data
    ]
    party_info = {"count": len(party_pokemon), "pokemon": party_pokemon}

    # --- 4. 아이템 정보 ---
    inventory = {"items": reader.read_items()}

    # --- 5. 이벤트 상태 (수정됨) ---
    # 새로운 함수를 호출하여 모든 이벤트 플래그를 동적으로 읽어옵니다.
    event_statuses = read_all_event_flags(pyboy)

    return {
        "location": location,
        "player_info": player_info,
        "party_info": party_info,
        "inventory": inventory,
        "event_statuses": event_statuses,
    }