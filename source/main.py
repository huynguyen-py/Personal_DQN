from lib import *
from agent import Agent

if __name__ == "__main__":
    # Khởi tạo môi trường, agent, scores, mảng lịch sử eps
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.001)
    scores, eps_history = [], []

    # Train 500 episode
    n_games = 500

    # Điểm thưởng cao nhất để làm cờ hiệu cho lưu trữ model
    best_score = -np.inf

    # tiến hành train 500 lần
    for i in range(n_games):
        score = 0
        # Biến hiển thị trò chơi kết thúc hay chưa
        done = False
        # Khởi tạo lại môi trường
        observation = env.reset()
        while not done:
            # Hiển thị giao diện trò chơi
            env.render()
            # Agent chọn action ngẫu nhiên hoặc chọn action (max)  thông qua mạng DQN
            action = agent.choose_action(observation)
            # Thực hiện action và nhận về các output tương ứng, lưu trữ cộng dồn điểm (phần thưởng)
            observation_, reward, done, info = env.step(action)
            score += reward
            # Lưu trữ cho kĩ thuật "experience replay"
            agent.store_transition(observation, action, reward, observation_, done)
            # Tiến hành học
            agent.learn()
            # Gán lại quan sát mới từ new state
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        # Chọn 100 phần tử trở lại đây để làm avg_score
        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'avarage score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

        # nếu điểm (reward) hơn bước tối ưu trước, lưu lại model ở episode này
        if avg_score >= best_score:
            # file weight được lưu trữ ở "/weights/dqn_model"
            agent.save_model('{}/dqn_model'.format("../weights"))
            best_score = avg_score



