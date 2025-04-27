Here’s the updated `README.md` file tailored in a similar fashion, with placeholders for your GitHub URL, author name, and institution. You can replace the placeholders with your specific details.

---

# 🧠 Slime Volleyball AI Agents: Minimax & Alpha-Beta Pruning

This project implements **Minimax** and **Alpha-Beta Pruning** algorithms on the classic **Slime Volleyball** environment using Python and OpenAI Gym. It also includes a **Random Agent** for baseline comparison and scripts to simulate, evaluate, and record gameplay.

---

## 📁 Repository Structure

```
AI_2/
├── algorithm/
│   ├── minimax.py             # Minimax implementation
│   ├── alphabeta.py           # Alpha-Beta Pruning implementation
│   └── random_agent.py        # Random agent for comparison
├── main.py                    # Main evaluation script
├── minimax_eval.py            # Runs only Minimax agent
├── alphabeta_eval.py          # Runs only Alpha-Beta agent
├── minimax_game.mp4           # Recorded gameplay (Minimax)
├── alphabeta_game.mp4         # Recorded gameplay (Alpha-Beta)
├── requirements.txt           # Python dependencies
└── README.md                  # Project description (this file)
```

---

## ⚙️ Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-github-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Run both agents and save full gameplay:
```bash
python main.py
```

### Run Minimax agent only:
```bash
python minimax_eval.py
```

### Run Alpha-Beta agent only:
```bash
python alphabeta_eval.py
```

Each script records a full **30-second match at 30 fps** and saves the video for visualization.

---

## 🧪 Evaluation

| Agent         | Score (Yellow vs. Blue) | Frames | Execution Time |
|---------------|--------------------------|--------|----------------|
| Minimax       | 0 - 10                   | 1200   | 266.04 s       |
| Alpha-Beta    | 0 - 10                   | 1200   | 123.68 s       |

- **Blue** was the **maximizing player**.
- **Alpha-Beta Pruning** provided identical performance to Minimax but with nearly **2x faster execution**.

---

## 🎯 Evaluation Function

Both algorithms used a **simple evaluation function** based on:
- Distance between the agent and the ball.
- Positional advantage on the field.

This heuristic allowed reasonable control and decision-making for the agents.

---

## 🔍 Key Insights

- **Minimax vs. Alpha-Beta:** Same results, Alpha-Beta is faster.
- **Random Agent:** No strategy, poor performance.
- **Evaluation:** Effective, but limited by simplistic features.

---

## 🚧 Challenges & Future Work

- **Challenges:**
  - Limited planning depth affects long-term strategies.
  - Difficulty modeling dynamic opponent and ball physics.

- **Future Work:**
  - Improve evaluation using ball velocity, opponent position, etc.
  - Integrate **reinforcement learning** for adaptive strategy learning.

---

## 📽️ Demo Videos

- [🎮 Minimax Gameplay](minimax_game.mp4)
- [🎮 Alpha-Beta Gameplay](alphabeta_game.mp4)

---

## 🤝 Credits

Developed as part of **AI Assignment 3** @ <your-institution>  
**Authors:**
ABHISHEK KUMAR (CS24M120)
SANDEEP KUMAR  (CS24M112)


---