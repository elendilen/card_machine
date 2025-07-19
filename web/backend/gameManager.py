from collections import defaultdict
import requests
import time
import threading
from speaker import Speaker

class GameManager:
    def __init__(self):
        self.state = {
            "phase": "night",      # night or day
            "round": 1,
            "current_role": None,  # 当前执行动作的角色
            "current_player": None,# 当前执行动作的玩家ID
            "players": {},
            "actions": {"kill": None, "save": None, "poison": None, "check": None},
            "dead_tonight": [],
            "seer_result": "",     # ✅ 预言家查验结果
            "witch_info": "",
            "winner": None,  # 游戏胜利方
        }
        self.role_order = ["werewolf", "seer", "witch"]  # 夜晚行动顺序
        self.current_index = 0
        self.bad_players = set()  # 坏人队列（狼人）
        self.wolves_known = False # 是否已识别所有狼人
        self.witch_step = None    # 女巫动作阶段
        self.players = self.state["players"]
        self.speaker = Speaker()

    
    def init_players(self, num_players):
        # 重置状态
        self.state["phase"] = "night"
        self.state["round"] = 1
        self.state["actions"] = {"kill": None, "save": None, "poison": None, "check": None}
        self.state["dead_tonight"] = []

        # 初始化玩家
        self.state["players"] = {i: {"role": "unknown", "alive": True} for i in range(1, num_players + 1)}

        # 设置当前执行角色（第一个行动是狼人）
        self.current_index = 0
        self.next_role()



    def assign_roles(self, roles):
        # roles = ["werewolf", "villager", "seer", "witch"]
        for i, role in enumerate(roles, start=1):
            self.state["players"][i] = {"role": role, "alive": True}
        self.next_role()  # 初始化到第一个角色

    def trigger_camera_capture(self):
        try:
            url = "http://127.0.0.1:5000/send_command"
            response = requests.post(url, params={"command": "rotate_90_and_capture"}, timeout=5)
            print(f"已发送拍照命令给硬件，响应: {response.json()}")
        except Exception as e:
            print(f"拍照命令失败: {e}")

    def next_role(self):
        self.state["current_prompt"] = ""
        self.state["seer_result"] = ""
        self.state["witch_info"] = ""
        if self.current_index < len(self.role_order):
            self.state["current_role"] = self.role_order[self.current_index]
            time.sleep(10)
            self.trigger_camera_capture()
            # 如果是狼人阶段且还未记录过狼人 → 记录所有狼人到坏人队列
            if self.state["current_role"] == "werewolf" and not self.wolves_known:
                for pid, info in self.state["players"].items():
                    if info["role"] == "werewolf":
                        self.bad_players.add(pid)
                self.wolves_known = True

            # 设置提示信息
            if self.state["current_role"] == "werewolf":
                self.state["current_prompt"] = "Night falls,everyone close your eyes. Werewolves, open your eyes and choose a player to kill"
                self.set_prompt("Night falls,everyone close your eyes. Werewolves, open your eyes and choose a player to kill")
            elif self.state["current_role"] == "seer":
                self.state["current_prompt"] = "Seer, open your eyes and choose a player to check their role"
                self.set_prompt("Seer, open your eyes and choose a player to check their role")
            elif self.state["current_role"] == "witch":
                self.witch_step = "save"
                self.state["witch_info"] = f"Player {self.state['actions']['kill']} was attacked last night. Save or poison?"
                self.set_prompt("Witch, open your eyes. Do you want to save the attacked player or poison someone?")
                self.delayed_clear("witch_info", delay=30)
                self.current_index += 1
                return  # 等待女巫操作
            
            target_player = None
            # 找到第一个活着的这个角色的玩家
            for pid, info in self.state["players"].items():
                if info["role"] == self.state["current_role"] and info["alive"]:
                    target_player = pid
                    break
                
            if target_player is not None:
                print(f"[⚠️] 所有 {self.state['current_role']} 玩家都已死亡，跳过该角色")
                self.current_index += 1
                self.next_role()  # ✅ 递归进入下一个角色
                return
            else:
                self.state["current_player"] = target_player
            self.current_index += 1
        else:
            # 所有夜晚行动结束，进入白天
            self.resolve_night()
            self.state["phase"] = "day"
            self.state["current_prompt"] = f"天亮了！昨晚死亡玩家：{self.state['dead_tonight'] or '无人'}"
            msg = f"Day has come. Players who died last night was: {', '.join(map(str, self.state['dead_tonight'])) if self.state['dead_tonight'] else 'No one'}"
            self.set_prompt(msg)
            self.check_game_end()
            self.current_index = 0

    def handle_action(self, gesture):
        role = self.state["current_role"]
        print(f"[🧠] 当前角色: {role}")
        if role == "werewolf":
            self.state["actions"]["kill"] = int(gesture)
            self.state["current_prompt"] = f"狼人选择击杀 {gesture} 号"
            self.next_role()
        elif role == "seer":
            target = int(gesture)
            self.state["actions"]["check"] = target
            print(f"[DEBUG] 当前坏人列表: {self.bad_players}")
            print(f"[DEBUG] 查验目标 {target} 的角色: {self.state['players'][target]['role']}")
            result = "bad guys" if self.state["players"][target]["role"] == "werewolf" else "good person"
            msg = f"Seer checked Player {target}: {result}"
            self.state["seer_result"] = msg
            self.state["current_prompt"] = msg  # ✅ 显示到屏幕上
            self.delayed_clear("seer_result", delay=10)
            threading.Thread(target=lambda: (time.sleep(8), self.next_role()), daemon=True).start()
        elif role == "witch":
            if gesture.lower() == "ok":
                self.state["actions"]["save"] = self.state["actions"]["kill"]
                self.state["current_prompt"] = "女巫选择救人，放弃毒药"
                print(f"[🧪] 女巫救下了 {self.state['actions']['kill']} 号，不使用毒药")
                
            elif gesture.lower() == "x":
                self.state["current_prompt"] = "女巫放弃救人和用毒"
                print("[🧪] 女巫放弃救人和使用毒药")

            else:
                # 默认只毒人，不救人
                try:
                    poison_id = int(gesture)
                    self.state["actions"]["poison"] = poison_id
                    self.state["current_prompt"] = f"女巫毒死了 {poison_id} 号"
                    print(f"[☠️] 女巫选择不救人，毒死：{poison_id}")
                except:
                    self.state["current_prompt"] = "女巫手势识别错误，跳过操作"
                    print("[⚠️] 女巫手势解析失败，跳过")

            self.next_role()
            


    def resolve_night(self):
        print("🌙 进入 resolve_night，结算夜晚行动")
        kill_target = self.state["actions"]["kill"]
        save_target = self.state["actions"]["save"]
        poison_target = self.state["actions"]["poison"]

        if kill_target and kill_target != save_target:
            self.state["players"][kill_target]["alive"] = False
            self.state["dead_tonight"].append(kill_target)
            print(f"🩸 狼人杀死了 {kill_target}（未被救）")
            
        if poison_target:
            self.state["players"][poison_target]["alive"] = False
            self.state["dead_tonight"].append(poison_target)
            print(f"☠️ 女巫毒死了 {poison_target}")
        self.state["actions"] = {"kill": None, "save": None, "poison": None, "check": None}
        self.state["current_role"] = "day"
        
        # ✅ 修复：确保投票状态初始化
        self.state["votes"] = defaultdict(int)
        self.state["voted_today"] = set()
        self.state["eliminated_today"] = None
        
    # 动态更新角色（通过人脸识别）
    def update_role(self, player_id: int):
        role = self.state["current_role"]
        if role in ["werewolf", "seer", "witch"]:
            self.state["players"][player_id]["role"] = role
            if role == "werewolf":
                self.bad_players.add(player_id)
            return True
        return False
    
    def handle_day_vote(self, voter_id, vote_target):
        if self.state["phase"] != "day":
            return False, "当前不是白天阶段"
        if not self.state["players"][voter_id]["alive"]:
            return False, "你已死亡，不能投票"
        if voter_id in self.state["voted_today"]:
            return False, "你已经投过票"

        self.state["votes"][vote_target] += 1
        self.state["voted_today"].add(voter_id)

        # 如果所有活人都投票了，结算结果
        alive_players = [pid for pid, info in self.state["players"].items() if info["alive"]]
        if len(self.state["voted_today"]) == len(alive_players):
            max_votes = max(self.state["votes"].values())
            candidates = [pid for pid, count in self.state["votes"].items() if count == max_votes]
            eliminated = candidates[0]  # 平票默认杀第一个
            self.state["players"][eliminated]["alive"] = False
            self.state["eliminated_today"] = eliminated
            self.state["current_prompt"] = f"投票结果：{eliminated} 号被放逐"
            self.check_game_end()
            if not self.state.get("winner"):  # 没结束才推进
                self.advance_phase()
        return True, "投票成功"

    def check_game_end(self):
        alive = [p for p, info in self.state["players"].items() if info["alive"]]
        wolves = [p for p in alive if self.state["players"][p]["role"] == "werewolf"]
        others = [p for p in alive if self.state["players"][p]["role"] != "werewolf"]

        if not wolves:
            self.state["winner"] = "好人阵营"
            self.state["current_prompt"] = "Game over: Villagers win"
            self.set_prompt("Game over: Villagers win")
            print("🏁 游戏结束，好人阵营胜利！")
            self._print_final_state()
        elif len(wolves) >= len(others):
            self.state["winner"] = "狼人阵营"
            self.state["current_prompt"] = "Game over: Werewolves win"
            self.set_prompt("Game over: Werewolves win")
            print("🏁 游戏结束，狼人阵营胜利！")
            self._print_final_state()

    def advance_phase(self):
        # 若游戏结束则不再继续
        if self.state["winner"]:
            return

        if self.state["phase"] == "day":
            # 开始新一轮夜晚
            self.state["phase"] = "night"
            self.state["round"] += 1
            self.state["actions"] = {"kill": None, "save": None, "poison": None, "check": None}
            self.state["dead_tonight"] = []
            self.state["votes"] = defaultdict(int)
            self.state["voted_today"] = set()
            self.state["eliminated_today"] = None
            self.state["current_player"] = None
            self.state["current_role"] = None
            self.current_index = 0
            # ✅ 添加这句，否则前端没有任何提示
            self.state["current_prompt"] = "夜晚来临，狼人请睁眼"
            
            print("☀️ 进入夜晚阶段")
            self.next_role()
        else:
            # 夜晚结束，进入白天
            self.state["phase"] = "day"
            self.state["voted_today"] = set()
            self.state["votes"] = {pid: 0 for pid in self.state["players"] if self.state["players"][pid]["alive"]}
            self.state["current_role"] = "day"
            self.state["current_prompt"] = "☀️ 天亮了，请开始发言并投票"
            print("☀️ 进入白天阶段")
            self.state["current_prompt"] = "请等待夜晚阶段完成后自动进入白天"


    def run_game_step(self, input_data=None):
        phase = self.state["phase"]
        current_role = self.state["current_role"]

        if self.state["winner"]:
            return "游戏已结束"

        if phase == "night":
            # 模拟输入动作（真实系统中这部分已由 /submit_gesture 驱动）
            if input_data:
                self.handle_action(input_data)  # 提交手势
            if self.current_index >= len(self.role_order):
                self.resolve_night()
                self.state["phase"] = "day"
                self.state["current_prompt"] = f"天亮了，昨晚死亡玩家：{self.state['dead_tonight'] or '无人'}"
                self.current_index = 0
        elif phase == "day":
            if input_data and isinstance(input_data, tuple):  # (voter_id, vote_target)
                voter_id, vote_target = input_data
                self.handle_day_vote(voter_id, vote_target)

            # 判断投票是否完成在 handle_day_vote 中
        return self.state["current_prompt"]
    
    def werewolf_kill(self, target_id: int):
        if self.state != "night":
            return "当前不是狼人行动阶段"

        self.kill_target = target_id
        return f"已标记玩家 {target_id} 为狼人目标"
    
    def start_night(self):
        print("🌃 新一轮夜晚开始")
        self.state["current_role"] = "werewolf"
        self.state["actions"] = {"kill": None, "save": None, "poison": None, "check": None}
        self.state["dead_tonight"] = []

    def eliminate_player(self, player_id: int):
        if player_id not in self.state["players"]:
            return False, f"玩家 {player_id} 不存在"

        if not self.state["players"][player_id]["alive"]:
            return False, f"玩家 {player_id} 已死亡"

        self.state["players"][player_id]["alive"] = False
        self.state["eliminated_today"] = player_id
        return True, f"玩家 {player_id} 被处决"

    def _print_final_state(self):
        print("🎯 最终玩家状态：")
        for pid, info in self.state["players"].items():
            status = "存活" if info["alive"] else "死亡"
            print(f"玩家 {pid} - 角色: {info['role']}, 状态: {status}")
        print(f"🎉 获胜方：{self.state['winner']}")

    def delayed_clear(self, field, delay=20):
        def clear():
            time.sleep(delay)
            self.state[field] = ""
            print(f"[🧹] 延迟清空字段 {field}")
        threading.Thread(target=clear, daemon=True).start()

    def set_prompt(self, msg: str):
        self.state["current_prompt"] = msg
        self.speaker.say(msg)
