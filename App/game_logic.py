import random
import json
import os

class AI:
    def __init__(self, name, health, attack, defense, speed, q_table_file='q_table.json', 
                 learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2, can_heal = True):
        self.name = name
        self.health = health
        self.attack = attack
        self.defense = defense
        self.speed = speed
        self.can_heal = can_heal  # 체력 회복 가능 여부
        self.q_table_file = q_table_file
        
        # 강화학습 파라미터
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_table = self.load_q_table()

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Failed to load Q-table from {self.q_table_file}. Initializing empty Q-table.")
                return {}
        else:
            return {}

    def save_q_table(self):
        try:
            with open(self.q_table_file, 'w', encoding='utf-8') as f:
                json.dump(self.q_table, f, ensure_ascii=False, indent=4)
            print(f"Q-table saved to {self.q_table_file}")
        except IOError as e:
            print(f"Error saving Q-table: {e}")

    def choose_action(self, state):
        """상태에 따라 AI의 행동을 선택합니다."""
        actions = ['공격', '방어', '회복']
        if random.random() < self.exploration_rate:  # 탐험(exploration) 확률
            action = random.choice(actions)
            print(f"{self.name} is exploring: Chose random action: {action}")
            return action
        else:
            # Q-값을 사용해 행동 선택 (가장 큰 Q-값을 가진 행동 선택)
            q_values = [self.q_table.get(f"{state}_{action}", 0) for action in actions]
            if all(q == 0 for q in q_values):  # Q-값이 모두 0인 경우, 무작위 선택
                action = random.choice(actions)
                print(f"{self.name} has no learned Q-values: Chose random action: {action}")
                return action
            action = actions[q_values.index(max(q_values))]
            print(f"{self.name} chose action: {action} (Q-value: {max(q_values)})")
            return action

    def update_q_value(self, action, reward, next_state):
        """Q-값을 업데이트합니다."""
        # 스테이지 제거하고 행동(action)만을 키로 사용
        key = f"{action}"  # 스테이지 정보 제거
        if key not in self.q_table:
            self.q_table[key] = 0  # Q-value 초기화

        # Q-learning 업데이트 공식
        max_q_value_next_state = self.get_max_q_value(next_state)
        self.q_table[key] = self.q_table[key] + self.learning_rate * (
            reward + self.discount_factor * max_q_value_next_state - self.q_table[key]
        )

    def get_max_q_value(self, state):
        """주어진 상태에서 가능한 행동들 중 최대 Q-값을 반환합니다."""
        actions = ['공격', '방어', '회복']
        q_values = [self.q_table.get(f"{state}_{action}", 0) for action in actions]
        return max(q_values)

    def take_damage(self, damage):
        """피해를 받는 메서드"""
        self.health -= max(0, damage)
        print(f"{self.name} takes {damage} damage. Health is now {self.health}.")

    def heal(self, amount):
        """회복 메서드"""
        self.health += amount
        print(f"{self.name} heals {amount}. Health is now {self.health}.")

    def is_alive(self):
        """생존 여부 확인"""
        return self.health > 0

class Character:
    def __init__(self, name, health, attack, defense, speed, can_heal = True):
        self.name = name
        self.health = health
        self.attack = attack
        self.defense = defense
        self.speed = speed
        self.can_heal = can_heal  # 체력 회복 가능 여부

    def __str__(self):
        return f"{self.name} (Health: {self.health}, Attack: {self.attack}, Defense: {self.defense}, Speed: {self.speed})"
    
    def choose_action(self, action):
        """플레이어가 선택한 행동을 반환"""
        return action  # Flask에서 폼으로 받은 행동

    def take_damage(self, damage):
        """피해를 받는 메서드"""
        self.health -= max(0, damage)

    def heal(self, amount):
        """회복 메서드"""
        self.health += amount

    def is_alive(self):
        """생존 여부 확인"""
        return self.health > 0


def generate_enemy(stage_level):
    stage_data = load_stage_data().get('stages', [])
    stage_info = next((stage for stage in stage_data if stage['stage'] == stage_level), None)
    
    if stage_info:
        enemy_stats = stage_info['enemy']
        return AI(
            name="Enemy",
            health=enemy_stats['hp'],
            attack=enemy_stats['attack'],
            defense=enemy_stats['defense'],
            speed=10
        )
    else:
        print(f"Stage {stage_level} data not found. Generating default enemy.")
        return AI(name="Enemy", health=50, attack=10, defense=5, speed=10)


def load_stage_data():
    """스테이지 데이터를 로드하는 함수"""
    json_file_path_stage = os.path.join(os.path.dirname(__file__), 'assets', 'stage_stats.json')
    with open(json_file_path_stage, 'r') as json_file:
        return json.load(json_file)


def battle_with_ai(player, enemy):
    """전투를 진행하는 함수"""
    state = '시작'  # 상태 초기화
    round_num = 1

    while player.is_alive() and enemy.is_alive():
        print(f"\nRound {round_num}:")
        
        # 플레이어 행동 선택
        player_action = player.choose_action(state)
        print(f"Player chooses action: {player_action}")
        
        # 적 AI의 행동 선택
        enemy_action = enemy.choose_action(state)
        print(f"{enemy.name} chooses action: {enemy_action}")
        
        # 플레이어와 적의 행동에 따른 전투 결과 적용
        # 1. 플레이어의 공격
        if player_action == '공격' and enemy_action != '방어':
            enemy.take_damage(player.attack)
            print(f"Player attacks {enemy.name}, causing {player.attack} damage.")
        elif player_action == '방어' and enemy_action == '공격':
            damage_taken = max(0, enemy.attack - player.defense)
            player.take_damage(damage_taken)
            print(f"Player defends against {enemy.name}'s attack, taking {damage_taken} damage.")
        
        # 2. AI의 공격
        if enemy_action == '공격' and player_action != '방어':
            player.take_damage(enemy.attack)
            print(f"{enemy.name} attacks Player, causing {enemy.attack} damage.")
        elif enemy_action == '방어' and player_action == '공격':
            damage_taken = max(0, player.attack - enemy.defense)
            enemy.take_damage(damage_taken)
            print(f"{enemy.name} defends against Player's attack, taking {damage_taken} damage.")
        
        # 상태 출력
        print(f"Player Health: {player.health}, {enemy.name} Health: {enemy.health}")
        
        # Q-테이블 업데이트
        # 플레이어와 AI가 상태에 따라 Q-값을 업데이트
        enemy.update_q_value('중간', enemy_action, 10, '끝')
        
        # 라운드 번호 증가
        round_num += 1
    
    # 전투 결과
    if player.is_alive():
        print(f"{player.name} wins!")
    else:
        print(f"{enemy.name} wins!")

    enemy.save_q_table()

def choose_player_type(type_name):
    """플레이어 타입에 맞는 캐릭터 설정 함수"""
    player_types = load_player_types()
    player_stats = next((player for player in player_types if player['type'] == type_name), None)
    
    if player_stats:
        return Character(
            name="Player",
            health=player_stats['stats']['hp'],
            attack=player_stats['stats']['attack'],
            defense=player_stats['stats']['defense'],
            speed=player_stats['stats']['speed']
        )
    else:
        return None

def load_player_types():
    json_file_path_player = os.path.join(os.path.dirname(__file__), 'assets', 'stat.json')
    try:
        with open(json_file_path_player, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            # 데이터 검증
            for player in data:
                if 'type' not in player or 'stats' not in player:
                    raise ValueError(f"Invalid player data: {player}")
            return data
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading player types: {e}")
        return []
