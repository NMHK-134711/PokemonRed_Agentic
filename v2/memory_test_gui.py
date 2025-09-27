# memory_test_gui.py (골드 버전 로직 적용 최종본)
import tkinter as tk
from tkinter import ttk
import json
from pyboy import PyBoy
from memory_reader import PokemonRedReader
import os

# --- 설정 ---
GB_PATH = os.path.abspath('../PokemonRed.gb')
INIT_STATE_PATH = os.path.abspath('../initial_red.state') 

class MemoryReaderGUI:
    def __init__(self, master):
        self.master = master
        master.title("Pokémon Red - Memory Inspector (Improved Logic)")
        master.geometry("600x500")

        # --- PyBoy 초기화 ---
        print("PyBoy를 초기화합니다... (게임 창이 별도로 뜹니다)")
        self.pyboy = PyBoy(
            GB_PATH,
            window="SDL2",
            sound=False,
        )
        self.pyboy.set_emulation_speed(0) # tick()으로 속도를 100% 제어
        with open(INIT_STATE_PATH, "rb") as f:
            self.pyboy.load_state(f)
        print("PyBoy 초기화 완료. 게임 창에서 직접 플레이하세요.")

        self.reader = PokemonRedReader(self.pyboy.memory)

        # --- GUI 위젯 생성 (이전과 동일) ---
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.vars = {
            "player_name": tk.StringVar(),
            "location": tk.StringVar(),
            "coords": tk.StringVar(),
            "badges": tk.StringVar(),
            "money": tk.StringVar(),
            "battle_status": tk.StringVar(),
        }

        row = 0
        for key, var in self.vars.items():
            label = ttk.Label(self.main_frame, text=f"{key.replace('_', ' ').title()}:", font=("Helvetica", 12, "bold"))
            label.grid(row=row, column=0, sticky="w", pady=2)
            value_label = ttk.Label(self.main_frame, textvariable=var, font=("Helvetica", 12))
            value_label.grid(row=row, column=1, sticky="w", pady=2)
            row += 1
            
        party_label = ttk.Label(self.main_frame, text="Party Info:", font=("Helvetica", 12, "bold"))
        party_label.grid(row=row, column=0, sticky="nw", pady=10)
        self.party_text = tk.Text(self.main_frame, height=100, width=60, font=("Courier", 10))
        self.party_text.grid(row=row, column=1, sticky="w", pady=10)

        # ############################################################### #
        # ## ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 수정된 부분 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ ##
        # ############################################################### #
        
        # 1. 게임 실행 루프를 시작합니다. (약 60 FPS)
        self.emulation_tick()
        # 2. GUI 업데이트 루프를 시작합니다. (10 Hz)
        self.update_gui()

    def emulation_tick(self):
        """오직 게임 에뮬레이터만 계속 실행시키는 역할 (약 60 FPS)"""
        self.pyboy.tick()
        # 16ms 뒤에 자기 자신을 다시 호출하여 루프를 지속
        self.master.after(16, self.emulation_tick)

    def update_gui(self):
        """메모리를 읽고 GUI를 업데이트하는 역할 (100ms 마다)"""
        try:
            player_name = self.reader.read_player_name()
            location = self.reader.read_location()
            x, y = self.reader.read_coordinates()
            badges_list = self.reader.read_badges()
            money = self.reader.read_money()
            party_pokemon = self.reader.read_party_pokemon()
            
            in_battle_flag = self.pyboy.memory[0xD057]
            battle_status = "In Battle" if in_battle_flag != 0 else "Not in Battle"

            self.vars["player_name"].set(player_name)
            self.vars["location"].set(location)
            self.vars["coords"].set(f"X: {x}, Y: {y}")
            self.vars["badges"].set(f"{len(badges_list)} badges ({', '.join(badges_list)})")
            self.vars["money"].set(f"${money}")
            self.vars["battle_status"].set(battle_status)

            party_list = []
            for p in party_pokemon:
                p_dict = p.__dict__
                p_dict['status'] = p.status.get_status_name()
                p_dict['type1'] = p.type1.name
                p_dict['type2'] = p.type2.name if p.type2 else None
                party_list.append(p_dict)

            party_info_str = json.dumps(party_list, indent=2)
            self.party_text.delete('1.0', tk.END)
            self.party_text.insert(tk.END, party_info_str)

        except Exception as e:
            self.party_text.delete('1.0', tk.END)
            self.party_text.insert(tk.END, f"An error occurred:\n{e}")

        # 100ms 뒤에 자기 자신을 다시 호출하여 루프를 지속
        self.master.after(100, self.update_gui)
        
        # ############################################################### #
        # ## ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 수정된 부분 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ##
        # ############################################################### #

if __name__ == "__main__":
    root = tk.Tk()
    app = MemoryReaderGUI(root)
    root.mainloop()