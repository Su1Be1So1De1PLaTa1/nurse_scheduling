from flask import Flask, jsonify, request, render_template, Response
import numpy as np
import matplotlib.pyplot as plt
import datetime
import jpholiday
import io
import base64
import time
import json

app = Flask(__name__)

def is_weekend(day):
    return day % 7 == 5 or day % 7 == 6  # 土曜日は5、日曜日は6（0から始まる）

def is_holiday(day):
    date = datetime.date(2024, 6, 1) + datetime.timedelta(days=day)
    return jpholiday.is_holiday(date)

num_days = 31
Q = None
alpha = 0.1
gamma = 0.99
epsilon = 0.3
state = None
reward_list = []
count = []

def reward_function(state, day, num_nurses, num_day_shift, num_night_shift):
    count_night = np.sum(state[:, day] == 2)
    count_day = np.sum(state[:, day] == 1)
    penalty = 0
    reward = 0

    count_after = np.sum(state[:, 0] == 3)
    if is_weekend(day) or is_holiday(day):
        if count_night == num_night_shift:
            reward += 30
        if count_day == num_day_shift:
            reward += 30
    else:
        if count_night == num_night_shift:
            reward += 30
        if count_day >= num_day_shift:
            reward += 30

    for nurse in range(num_nurses):
        if day == num_days - 1:
            nurse_work_days = np.sum(state[nurse, :] == 1) + np.sum(state[nurse, :] == 2) + np.sum(state[nurse, :] == 3)
            if nurse_work_days > 20:
                penalty += 10
            if nurse_work_days < 12:
                penalty += 10

            if nurse_work_days >= 12 and nurse_work_days <= 20:
                reward += 20

        if day > 3 and state[nurse, day - 3] != 0 and state[nurse, day - 2] != 0 and state[nurse, day - 1] != 0 and state[nurse, day] != 0:
            penalty += 10

        if day > 1 and state[nurse, day - 1] == 0 and state[nurse, day] == 0:
            penalty += 10

    if penalty == 0 and day == num_days - 1:
        reward += 100

    return reward - penalty

def cbf(i, reward):
    count.append(i)
    reward_list.append(reward)

def run_q_learning(num_nurses, num_day_shift, num_night_shift):
    global Q, state, reward_list, count
    Q = np.zeros((num_nurses * num_days, 4))
    reward_list = []
    count = []
    total_episodes = 30000
    for episode in range(total_episodes):
        state = np.zeros((num_nurses, num_days), dtype=int)
        if episode > 29000:
            epsilon = 0
        elif episode > 25000:
            epsilon = 0.1
        else:
            epsilon = 0.3
        all_reward = 0
        for day in range(num_days):
            daily_rewards = []
            for nurse in range(num_nurses):
                current_state = nurse * num_days + day
                if day > 0 and state[nurse, day - 1] == 2:
                    action = 3
                elif day > 0 and state[nurse, day - 1] == 3:
                    action = np.random.choice([0, 1])
                    action = np.argmax(Q[current_state])
                    if action in (2, 3):
                        sorted_indices = np.argsort(Q[current_state])
                        action = sorted_indices[-2] if len(sorted_indices) > 1 else sorted_indices[-1]
                        if action in (2, 3):
                            action = sorted_indices[-3]

                elif np.random.rand() < epsilon:
                    action = np.random.randint(0, 4)
                else:
                    action = np.argmax(Q[current_state])
                    if action == 3:
                        sorted_indices = np.argsort(Q[current_state])
                        action = sorted_indices[-2] if len(sorted_indices) > 1 else sorted_indices[-1]

                state[nurse, day] = action

            reward = reward_function(state, day, num_nurses, num_day_shift, num_night_shift)
            all_reward += reward
            daily_rewards.append(reward)
            for nurse in range(num_nurses):
                current_state = nurse * num_days + day
                action = state[nurse, day]
                next_state = nurse * num_days + (day + 1) % num_days if day < num_days - 1 else current_state
                Q[current_state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[current_state, action])
        cbf(episode, all_reward)
        progress = (episode + 1) / total_episodes * 100
        yield f"data: {json.dumps({'progress': progress})}\n\n"
        time.sleep(0.01)

    yield f"data: {json.dumps({'message': 'Training complete!'})}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    num_nurses = int(request.args.get('num_nurses', 10))
    num_day_shift = int(request.args.get('num_day_shift', 6))
    num_night_shift = int(request.args.get('num_night_shift', 1))
    return Response(run_q_learning(num_nurses, num_day_shift, num_night_shift), content_type='text/event-stream')

@app.route('/schedule')
def get_schedule():
    schedule = np.vectorize(lambda x: ['休み', '昼勤', '夜勤', '夜勤明け'][x])(state).tolist()
    return jsonify(schedule)

@app.route('/plot.png')
def plot():
    plt.plot(count, reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
