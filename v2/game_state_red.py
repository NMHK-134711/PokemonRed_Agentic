# game_state_red.py (Refactored)
from memory_reader import PokemonRedReader # 새로 추가된 memory_reader.py를 임포트합니다.

def get_game_state(pyboy) -> dict:
    """
    PokemonRedReader를 사용하여 주요 게임 정보를 추출하고,
    기존 Agentic AI 시스템이 요구하는 형식의 dictionary로 변환하여 반환합니다.
    """
    # PokemonRedReader 인스턴스 생성
    reader = PokemonRedReader(pyboy.memory)

    # --- 1. 위치 정보 ---
    x_coord, y_coord = reader.read_coordinates()
    map_id = pyboy.memory[0xD35E] # map_id는 map_data.json과 연동되므로 직접 읽습니다.
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
    # PokemonData 객체 리스트를 LLM이 사용하기 쉬운 dict 리스트로 변환
    party_data = reader.read_party_pokemon()
    party_pokemon = [
        {"species_name": p.species_name, "level": p.level} for p in party_data
    ]
    party_info = {"count": len(party_pokemon), "pokemon": party_pokemon}

    # --- 4. 아이템 정보 ---
    # reader.read_items()는 이미 (이름, 수량) 튜플 리스트를 반환하므로 그대로 사용
    inventory = {"items": reader.read_items()}

    # --- 5. 이벤트 상태 ---
    # 이 정보는 PokemonRedReader에 없으므로 기존 방식을 유지합니다.
    event_statuses = {
        'defeated_lorelei': (pyboy.memory[0xD863] & (1 << 1)) != 0,
        'defeated_bruno': (pyboy.memory[0xD864] & (1 << 1)) != 0,
        'defeated_agatha': (pyboy.memory[0xD865] & (1 << 1)) != 0,
        'defeated_lance': (pyboy.memory[0xD866] & (1 << 6)) != 0,
        'became_champion': (pyboy.memory[0xD747] & (1 << 3)) != 0
    }

    # 최종적으로 기존과 동일한 구조의 딕셔너리를 반환합니다.
    return {
        "location": location,
        "player_info": player_info,
        "party_info": party_info,
        "inventory": inventory,
        "event_statuses": event_statuses,
    }