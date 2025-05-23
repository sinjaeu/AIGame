# AIGame

## 소개
AIGame은 Flask 기반의 웹 애플리케이션으로, 다양한 미니게임(미로찾기, 가위바위보, 로그라이크 전투 등)과 AI(강화학습 기반)와의 대결을 통해 재미와 학습을 동시에 경험할 수 있는 프로젝트입니다. AI는 Q-learning을 활용하여 플레이어와의 상호작용을 통해 점점 더 똑똑해집니다.

---

## 주요 기능
- **미로 게임**: Q-learning을 활용한 AI가 미로를 탐험하며 최적의 경로를 학습합니다.
- **가위바위보**: AI가 Q-learning으로 승률을 높이기 위해 학습합니다.
- **로그라이크 전투**: 플레이어와 AI가 각각 캐릭터를 선택해 전투를 벌입니다. AI는 Q-테이블을 기반으로 행동을 선택합니다.
- **캐릭터 선택**: 공격형, 방어형, 복합형 등 다양한 캐릭터 타입을 선택할 수 있습니다.
- **스테이지 시스템**: stage_stats.json에 정의된 여러 스테이지에서 점점 강해지는 적과 대결합니다.

---

## 폴더 구조
```
AIGame-master/
├── app.py                # 메인 Flask 앱
├── q_table.json          # AI의 Q-테이블 저장 파일
├── App/
│   ├── game_logic.py     # 게임 및 AI 로직
│   ├── assets/
│   │   ├── stage_stats.json  # 스테이지별 데이터
│   │   └── stat.json         # 캐릭터 타입별 스탯
├── static/
│   └── images/           # 이미지 등 정적 파일
├── templates/            # HTML 템플릿
│   ├── battle.html
│   ├── character_select.html
│   ├── main.html
│   ├── maze.html
│   ├── rogue.html
│   └── rps.html
```

---

## 실행 방법
1. **Python 3.8+ 설치 필요**
2. **필수 패키지 설치**
   ```bash
   pip install flask numpy matplotlib
   ```
3. **앱 실행**
   ```bash
   python app.py
   ```
4. **웹 브라우저에서 접속**
   - 기본 주소: [http://localhost:5000](http://localhost:5000)

---

## 주요 파일 설명
- **app.py**: Flask 서버 및 라우팅, 게임 진입점
- **App/game_logic.py**: AI, 캐릭터, 전투, 스테이지 등 게임의 핵심 로직
- **q_table.json**: AI의 학습 결과(Q-테이블)가 저장되는 파일
- **App/assets/stage_stats.json**: 각 스테이지별 플레이어/적의 능력치 데이터
- **App/assets/stat.json**: 캐릭터 타입별 기본 스탯
- **templates/**: 각 게임 모드별 HTML 템플릿

---

## AI 및 강화학습 설명
- **Q-learning**: AI는 Q-테이블을 통해 상태별 최적 행동을 학습합니다.
- **예시 (q_table.json)**
  ```json
  {
    "회복": -15.40,
    "방어": -10.83,
    "공격": -10.48
  }
  ```
- **스테이지 데이터 예시 (App/assets/stage_stats.json)**
  ```json
  {
    "stage": 1,
    "player": {"hp": 5, "attack": 2, "defense": 0},
    "enemy": {"hp": 50, "attack": 10, "defense": 3}
  }
  ```
- **캐릭터 타입 예시 (App/assets/stat.json)**
  ```json
  {
    "type": "공격형",
    "stats": {"hp": 70, "attack": 20, "defense": 5, "speed": 15}
  }
  ```

---

## 의존성 목록 (requirements.txt 예시)
```
flask
numpy
matplotlib
```

---

## 기타 참고사항
- 한글 폰트가 필요할 수 있습니다. (예: C:\Windows\Fonts\HANBatang.ttf)
- Q-테이블, 스테이지 데이터 등은 게임 진행에 따라 자동으로 갱신/저장됩니다.
- 추가적인 게임 모드나 AI 개선은 자유롭게 확장 가능합니다.

---

## 문의 및 기여
- Pull Request, Issue 환영합니다!
- 질문/건의: [프로젝트 관리자에게 문의] 
