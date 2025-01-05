import base64
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import random
from collections import deque
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from io import BytesIO
from App.game_logic import generate_enemy, Character, choose_player_type, AI, load_stage_data
from flask import session

# 한글 폰트 경로 설정
font_path = 'C:\\Windows\\Fonts\\HANBatang.ttf'  # 폰트 경로
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()  # 폰트 적용

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key')  # 환경 변수에서 가져오기

current_ai = None
stage_level = 1  # 처음 스테이지는 1로 시작

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.start_pos = (0, 0)  # 시작점
        self.goal_pos = (len(maze) - 1, len(maze[0]) - 1)  # 목표점
        self.state = self.start_pos  # 초기 상태는 시작점
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우
        self.q_table = np.zeros((len(maze), len(maze[0]), len(self.actions)))  # Q-table 초기화
        self.learning_rate = 0.2  # 학습률
        self.discount_factor = 0.9  # 할인율
        self.exploration_rate = 1.0  # 탐험률
        self.exploration_decay = 0.995  # 탐험률 감소율
        self.min_exploration_rate = 0.01  # 최소 탐험률
        self.max_distance = len(maze) * len(maze[0])  # 최대 거리 (최대 칸 수)

        # 추가: 거리 추적을 위한 변수 초기화
        self.previous_distance_to_goal = self.manhattan_distance(self.state, self.goal_pos)
        self.current_distance_to_goal = self.previous_distance_to_goal
        self.no_progress_count = 0  # 진전이 없을 경우 카운트

    def reset(self):
        self.state = self.start_pos  # 초기 상태로 리셋
        self.previous_distance_to_goal = self.manhattan_distance(self.state, self.goal_pos)  # 거리 초기화
        self.current_distance_to_goal = self.previous_distance_to_goal
        return self.state

    def is_valid_move(self, pos):
        x, y = pos
        return 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]) and self.maze[x][y] == 0

    def step(self, action):
        x, y = self.state
        dx, dy = self.actions[action]
        new_state = (x + dx, y + dy)

        if not self.is_valid_move(new_state):
            new_state = self.state  # 유효하지 않으면 원래 상태로 돌아옴

        reward = -1  # 기본 보상: -1 (벗어나는 경로)
        if new_state == self.goal_pos:
            reward = 100  # 목표에 도달하면 보상 +100

        # 목표와의 최소 거리 차이 계산
        self.previous_distance_to_goal = self.current_distance_to_goal
        self.current_distance_to_goal = self.manhattan_distance(new_state, self.goal_pos)

        # 보상 함수 수정: 목표에 가까워지면 보상 증가
        reward -= (self.previous_distance_to_goal - self.current_distance_to_goal)

        self.state = new_state
        return new_state, reward

    def manhattan_distance(self, pos1, pos2):
        # 맨해튼 거리 계산 (직선 거리가 아니라 수평, 수직으로만 이동)
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def bfs_check_path(self):
        """BFS로 시작점에서 목표까지 경로가 있는지 확인"""
        queue = deque([self.start_pos])
        visited = set()
        visited.add(self.start_pos)

        while queue:
            current_pos = queue.popleft()

            if current_pos == self.goal_pos:
                return True  # 경로가 있으면 True 반환

            for action in self.actions:
                next_pos = (current_pos[0] + action[0], current_pos[1] + action[1])

                if self.is_valid_move(next_pos) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)

        return False  # 경로가 없으면 False 반환

    def train(self, episodes=1000):
        if not self.bfs_check_path():  # 경로가 없다면 학습하지 않고 빈 리스트 반환
            return [], 0
        
        all_paths = []  # 모든 시도에서의 경로를 저장할 리스트
        found_path = False  # 경로가 존재하는지 여부를 나타내는 플래그
        for episode in range(episodes):
            state = self.reset()
            done = False
            episode_path = []  # 각 시도의 경로 기록
            while not done:
                # 탐험 또는 이용
                if random.uniform(0, 1) < self.exploration_rate:
                    action = random.randint(0, len(self.actions) - 1)  # 무작위로 액션 선택
                else:
                    # Q-value가 가장 큰 액션 선택
                    action = np.argmax(self.q_table[state[0], state[1]])

                # 선택한 액션에 대해 결과 얻기
                next_state, reward = self.step(action)
                episode_path.append(state)  # 상태를 경로에 추가

                # Q-value 업데이트
                old_q_value = self.q_table[state[0], state[1], action]
                future_q_value = np.max(self.q_table[next_state[0], next_state[1]])
                new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * future_q_value - old_q_value)
                self.q_table[state[0], state[1], action] = new_q_value

                state = next_state
                if state == self.goal_pos:  # 목표에 도달하면 종료
                    done = True

            # 목표에 도달하지 않았으면 진전이 없다고 판단하고 카운트 증가
            if self.current_distance_to_goal == self.previous_distance_to_goal:
                self.no_progress_count += 1
            else:
                self.no_progress_count = 0  # 진전이 있으면 카운트 초기화

            # 탐험률 감소
            if self.exploration_rate > self.min_exploration_rate:
                self.exploration_rate *= self.exploration_decay
            
            all_paths.append(episode_path)  # 시도별 경로 저장

            # 목표에 도달한 경로가 있으면 경로 존재로 표시
            if done:
                found_path = True

        # 경로 존재 여부 반환
        if found_path:
            return all_paths, episodes
        
import numpy as np
import random

import random
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class RPSAI:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.actions = ['rock', 'paper', 'scissors']
        self.q_table = np.zeros((3, 3))  # Q-테이블 초기화
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.previous_state = None
        self.previous_action = None
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_games = 0  # 총 게임 횟수
        self.win_rates = []  # 승률 기록을 위한 리스트

    def reset(self):
        """게임 초기화"""
        self.previous_state = None
        self.previous_action = None
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_games = 0  # 총 게임 횟수
        self.win_rates = []  # 승률 기록 초기화

    def choose_action(self, state):
        """행동 선택"""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            action_idx = np.argmax(self.q_table[state])
            return self.actions[action_idx]

    def get_reward(self, user_choice, ai_choice):
        """보상 계산"""
        if user_choice == ai_choice:
            self.draws += 1
            return 0
        elif (user_choice == 'rock' and ai_choice == 'scissors') or \
             (user_choice == 'paper' and ai_choice == 'rock') or \
             (user_choice == 'scissors' and ai_choice == 'paper'):
            self.losses += 1
            return -1
        else:
            self.wins += 1
            return 1

    def update_q_table(self, state, action, reward, next_state):
        """Q 테이블 업데이트"""
        action_idx = self.actions.index(action)
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action_idx] += self.alpha * (
            reward + self.gamma * self.q_table[next_state, best_next_action]
            - self.q_table[state, action_idx]
        )

    def play_round(self, user_choice):
        """한 라운드 플레이"""
        state = self.actions.index(user_choice)
        ai_choice = self.choose_action(state)

        reward = self.get_reward(user_choice, ai_choice)  # AI 기준 보상
        result = determine_winner(user_choice, ai_choice)  # 사용자 기준 결과

        if self.previous_state is not None:
            self.update_q_table(self.previous_state, self.previous_action, reward, state)

        self.previous_state = state
        self.previous_action = ai_choice

        self.total_games += 1
        win_rate = self.losses / self.total_games  # AI 기준 승률 계산
        self.win_rates.append(win_rate * 100)  # 승률 기록

        # epsilon 값 감소
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return ai_choice, reward, result  # 세 개의 값 반환

    def generate_win_rate_data(self):
        """승률 데이터를 반환하는 메서드"""
        games = list(range(1, self.total_games + 1))  # 시도 횟수에 대한 리스트
        win_rate = self.win_rates  # 승률 리스트

        return {
            'labels': games,  # x축 값 (게임 횟수)
            'data': win_rate   # y축 값 (승률)
        }

    def generate_action_data(self):
        """AI 선택 행동 데이터를 반환하는 메서드"""
        # Q-테이블 값에 대한 데이터를 반환
        actions = ['rock', 'paper', 'scissors']
        action_data = {
            'labels': actions,
            'data': self.q_table[:, 0].tolist()  # 예시로 첫 번째 열만 데이터로 사용 (각 선택에 대한 Q-Value)
        }
        return action_data

    def generate_win_rate_graph(self):
        """승률 그래프 생성"""
        games = list(range(1, self.total_games + 1))  # 시도 횟수에 대한 리스트
        
        # 승률을 계산해서 저장된 리스트를 바탕으로 그래프 그리기
        plt.switch_backend('Agg')  # 서버 환경에서 이미지 렌더링을 위한 백엔드 설정
        plt.figure(figsize=(6, 4))
        plt.plot(games, self.win_rates, marker='o', color='b', label='Win Rate')
        plt.title('승률 추세')
        plt.xlabel('게임 횟수')
        plt.ylabel('승률 (%)')
        plt.grid(True)
        
        # 이미지를 BytesIO 객체에 저장
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)  # 파일 포인터를 처음으로 이동
        plt.close()  # 그래프 종료

        # 이미지를 base64로 인코딩
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"  # base64로 변환한 이미지를 반환

rps_ai = RPSAI()

@app.route('/start_exploration', methods=['POST'])
def start_exploration():
    data = request.get_json()
    maze = data['maze']
    
    # 강화 학습 환경 생성 및 탐색 시작
    env = MazeEnv(maze)
    result = env.train()  # 탐색 경로와 시도 횟수 반환
    
    if result:
        path, attempt_count = result
        if path:
            return jsonify({'path': path, 'attemptCount': attempt_count})
        else:
            return jsonify({'path': None, 'attemptCount': attempt_count, 'message': 'No path found.'})
    else:
        return jsonify({'path': None, 'attemptCount': 0, 'message': 'Error in training. Could not find a path.'})
    
@app.route('/play', methods=['POST'])
def play():
    user_choice = request.form['choice']
    ai_choice, reward, result = rps_ai.play_round(user_choice)  # 사용자 기준 결과 추가

    # AI 학습 데이터와 사용자 승률 데이터 생성
    action_data = generate_action_data(rps_ai.q_table, ai_choice)
    win_rate_data = rps_ai.generate_win_rate_data()

    return jsonify({
        'user_choice': user_choice,
        'ai_choice': ai_choice,
        'result': result,  # 사용자 기준 결과
        'action_data': action_data,
        'win_rate_data': win_rate_data,
        'wins': rps_ai.losses,  # AI 기준 기록이 아니라 클라이언트에서 이해하는 승/패/무 기록 필요
        'losses': rps_ai.wins,
        'draws': rps_ai.draws
    })


def generate_action_data(q_table, ai_choice):
    """Generates data for the AI's action based on its Q-table"""
    actions = ['rock', 'paper', 'scissors']
    ai_choice_index = actions.index(ai_choice)
    q_values = q_table[ai_choice_index].tolist()  # Convert ndarray to list

    return {
        'labels': actions,
        'data': q_values  # Now it is a list, not an ndarray
    }


def determine_winner(user, ai):
    """사용자 기준으로 승패 판단"""
    if user == ai:
        return 'draw'
    elif (user == 'rock' and ai == 'scissors') or \
         (user == 'paper' and ai == 'rock') or \
         (user == 'scissors' and ai == 'paper'):
        return 'win'  # 사용자가 이겼을 경우
    else:
        return 'lose'  # 사용자가 졌을 경우

@app.route('/rps-reset', methods=['POST'])
def reset():
    rps_ai.reset()
    return jsonify({'status': 'AI의 상태가 초기화되었습니다.'})

@app.route('/maze-game')
def home():
    return render_template('maze.html')

@app.route('/rock-paper-scissors')
def rock_paper_scissors():
    return render_template('rps.html')

@app.route('/roguelike')
def roguelike():
    """로그라이크 메인 페이지"""
    return render_template('rogue.html')

@app.route('/character_select', methods=['GET', 'POST'])
def character_select():
    if request.method == 'POST':
        player_type = request.form.get('player_type')
        player = choose_player_type(player_type)
        
        # 세션에 플레이어 초기 상태 저장
        session['player'] = {
            "name": player.name,
            "health": player.health,
            "attack": player.attack,
            "defense": player.defense,
            "speed": player.speed
        }
        return redirect(url_for('start_stage', player_type=player_type))

    # GET 요청 시 캐릭터 선택 화면 렌더링
    return render_template('character_select.html')

@app.route('/start_stage', methods=['GET'])
def start_stage():
    # URL 파라미터로부터 player_type 가져오기
    player_type = request.args.get('player_type')
    if not player_type:
        return "Player type not provided. Please restart the game.", 400

    if 'player' not in session:
        return "Player data not found in session. Please restart the game.", 400

    # 초기 스테이지 설정
    stage_level = 1
    session['stage_level'] = stage_level  # 세션에 스테이지 저장

    # 세션에서 플레이어 상태 불러오기
    player_data = session['player']
    player = Character(
        name=player_data['name'],
        health=player_data['health'],
        attack=player_data['attack'],
        defense=player_data['defense'],
        speed=player_data['speed']
    )

    # 적 생성 및 세션에 저장
    enemy = generate_enemy(stage_level)
    session['enemy'] = {
        "name": enemy.name,
        "health": enemy.health,
        "attack": enemy.attack,
        "defense": enemy.defense,
        "speed": enemy.speed,
        "q_table_file": enemy.q_table_file
    }

    # 디버깅 로그 출력
    print(f"Starting stage {stage_level} with player type: {player_type}")
    print(f"Enemy stored in session: {session['enemy']}")

    # 템플릿 렌더링
    return render_template(
        'battle.html',
        player=player,
        enemy=enemy,
        stage_level=stage_level,
        player_type=player_type  # 템플릿으로 player_type 전달
    )

@app.route('/choose_action', methods=['POST'])
def choose_action():
    data = request.json
    player_action = data.get('action')
    stage_level = int(data.get('stage_level'))

    if 'player' not in session or 'enemy' not in session:
        return jsonify({"error": "Session expired. Restart the game."}), 400

    player_data = session['player']
    enemy_data = session['enemy']

    player = Character(
        name=player_data['name'],
        health=player_data['health'],
        attack=player_data['attack'],
        defense=player_data['defense'],
        speed=player_data['speed']
    )
    player.can_heal = player_data.get('can_heal', True)

    enemy = AI(
        name=enemy_data['name'],
        health=enemy_data['health'],
        attack=enemy_data['attack'],
        defense=enemy_data['defense'],
        speed=enemy_data['speed'],
        q_table_file=enemy_data.get('q_table_file', 'q_table.json')
    )
    enemy.can_heal = enemy_data.get('can_heal', True)

    enemy_action = enemy.choose_action('중간')
    round_result, reward = battle_round(player, enemy, player_action, enemy_action, stage_level)

    # Q-Values and probabilities
    q_values = [enemy.q_table.get(action, 0) for action in ['공격', '방어', '회복']]
    total_q = abs(sum(q_values))
    probabilities = [(abs(q) / total_q) * 100 if total_q > 0 else 0 for q in q_values]

    if not player.is_alive():
        return jsonify({"error": "플레이어가 사망했습니다..."}), 400
    elif not enemy.is_alive():
        player, enemy = next_stage(stage_level, reward)
        if not player or not enemy:
            return jsonify({"error": "모든 스테이지를 클리어하셨습니다!"}), 400

        stage_level += 1
        session['player'] = {
            "name": player.name,
            "health": player.health,
            "attack": player.attack,
            "defense": player.defense,
            "speed": player.speed,
            "can_heal": player.can_heal
        }
        session['enemy'] = {
            "name": enemy.name,
            "health": enemy.health,
            "attack": enemy.attack,
            "defense": enemy.defense,
            "speed": enemy.speed,
            "q_table_file": enemy.q_table_file,
            "can_heal": enemy.can_heal
        }
        session['stage_level'] = stage_level

        return jsonify({
            "message": f"Proceeding to stage {stage_level}",
            "stage_level": stage_level
        })

    session['player'] = {
        "name": player.name,
        "health": player.health,
        "attack": player.attack,
        "defense": player.defense,
        "speed": player.speed,
        "can_heal": player.can_heal
    }
    session['enemy'] = {
        "name": enemy.name,
        "health": enemy.health,
        "attack": enemy.attack,
        "defense": enemy.defense,
        "speed": enemy.speed,
        "q_table_file": enemy.q_table_file,
        "can_heal": enemy.can_heal
    }

    return jsonify({
        "player_action": player_action,
        "enemy_action": enemy_action,
        "player_health": player.health,
        "enemy_health": enemy.health,
        "round_result": round_result,
        "q_values": q_values,
        "probabilities": probabilities,
        "is_battle_over": not player.is_alive() or not enemy.is_alive()
    })

def battle_round(player, enemy, player_action, enemy_action, stage):
    """한 라운드 전투 처리"""
    round_result = ""

    # 플레이어와 적의 임시 방어력 증가량
    player_temp_defense = 0
    enemy_temp_defense = 0

    # 방어 행동 처리
    if player_action == '방어':
        player_temp_defense = 10  # 방어력 추가 증가 (조정 가능)
        round_result += f"플레이어가 방어를 선택하여 방어력이 {player_temp_defense}만큼 증가했습니다.\n"
    if enemy_action == '방어':
        enemy_temp_defense = 10
        round_result += f"{enemy.name}가 방어를 선택하여 방어력이 {enemy_temp_defense}만큼 증가했습니다.\n"

    # 플레이어의 행동 처리
    if player_action == '공격' and enemy_action != '방어':
        enemy.take_damage(player.attack)
        round_result += f"플레이어가 {enemy.name}에게 {player.attack}의 피해를 입혔습니다.\n"
    elif player_action == '공격' and enemy_action == '방어':
        damage_taken = max(0, player.attack - (enemy.defense + enemy_temp_defense))
        enemy.take_damage(damage_taken)
        round_result += f"{enemy.name}가 방어를 선택하여 {damage_taken}의 피해를 받았습니다.\n"
    elif player_action == '회복':
        if player.can_heal:
            player.heal(50)
            player.can_heal = False  # 회복 사용 불가로 설정
            round_result += f"플레이어가 체력을 50 회복했습니다. 현재 체력: {player.health}\n"
        else:
            round_result += "플레이어는 이미 회복을 사용했습니다. 회복할 수 없습니다.\n"

    # 적 AI의 행동 처리
    if enemy_action == '공격' and player_action != '방어':
        player.take_damage(enemy.attack)
        round_result += f"{enemy.name}가 플레이어에게 {enemy.attack}의 피해를 입혔습니다.\n"
    elif enemy_action == '공격' and player_action == '방어':
        damage_taken = max(0, enemy.attack - (player.defense + player_temp_defense))
        player.take_damage(damage_taken)
        round_result += f"플레이어가 방어를 선택하여 {damage_taken}의 피해를 받았습니다.\n"
    elif enemy_action == '회복':
        if enemy.can_heal:
            enemy.heal(50)
            enemy.can_heal = False  # 회복 사용 불가로 설정
            round_result += f"{enemy.name}가 체력을 50 회복했습니다. 현재 체력: {enemy.health}\n"
        else:
            round_result += f"{enemy.name}는 이미 회복을 사용했습니다. 회복할 수 없습니다.\n"
    
    # 전투 중 리워드 계산
    reward = calculate_reward(player, enemy, enemy_action)  # AI의 행동에 따른 리워드 계산
    enemy.update_q_value(enemy_action, reward, stage + 1)
    return round_result, reward

def next_stage(current_stage, reward):
    """다음 스테이지 데이터를 로드하고 플레이어 상태를 누적, 적 상태 초기화"""
    stage_data = load_stage_data()
    stages = stage_data.get('stages', [])
    next_stage_data = next((stage for stage in stages if stage['stage'] == current_stage + 1), None)
    
    if not next_stage_data:
        print("No more stages. Game completed!")
        return None, None  # 더 이상 스테이지가 없으면 종료

    # 다음 스테이지의 적 데이터
    enemy_data = next_stage_data['enemy']

    # 세션에서 플레이어 상태 가져오기
    player_data = session.get('player', {
        "health": 100,
        "attack": 10,
        "defense": 5,
        "speed": 10,
        "can_heal": True
    })

    # 플레이어 상태 누적
    player = Character(
        name="Player",
        health=player_data['health'] + next_stage_data['player']['hp'],  # 누적된 체력 유지
        attack=player_data['attack'] + next_stage_data['player']['attack'],  # 공격력 누적
        defense=player_data['defense'] + next_stage_data['player']['defense'],  # 방어력 누적
        speed=player_data['speed']  # 속도는 고정
    )
    player.can_heal = True

    # 새로운 적 생성
    enemy = AI(
        name="Enemy",
        health=enemy_data['hp'],
        attack=enemy_data['attack'],
        defense=enemy_data['defense'],
        speed=10  # 고정 속도
    )

    # Q-테이블 업데이트
    current_state = f"stage_{current_stage}"
    next_state = f"stage_{current_stage + 1}"
    action = enemy.choose_action(current_state)
    enemy.update_q_value(action, reward, next_state)
    enemy.save_q_table()

    return player, enemy

def calculate_reward(player, enemy, action):
    """AI가 승리하려고 노력하도록 보상 설계"""
    health_diff = enemy.health - player.health  # AI 체력과 플레이어 체력의 차이 계산

    if not player.is_alive() and enemy.is_alive():  # AI가 승리한 경우
        if action == '공격':
            reward = 20  # 승리 시 높은 보상
        elif action == '방어':
            reward = 15  # 방어도 승리에 기여하면 보상
        elif action == '회복':
            reward = 10  # 회복 시 승리에 가까워졌을 경우 보상
        else:
            reward = 5  # 승리하는 행동을 했으므로 기본 보상
    elif player.is_alive() and not enemy.is_alive():  # 플레이어가 승리한 경우
        reward = -20  # 패배 시 강한 부정적인 보상 (패배를 피하게 유도)
    else:  # 전투가 아직 끝나지 않은 경우 (중립적인 행동)
        if action == '공격':
            reward = max(0, health_diff)  # 공격 시 AI 체력 차이에 비례한 보상
        elif action == '방어':
            reward = max(0, 5 * (health_diff / 100))  # 방어 시 체력 차이에 따라 작은 보상
        elif action == '회복':
            reward = max(0, 10 * (1 - (enemy.health / (player.health + enemy.health))))  # 회복 시 상대 체력 비율로 보상
        else:
            reward = 0  # 무의미한 행동에 대해서는 보상 없음

    return reward

@app.route('/')
def main_page():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)
