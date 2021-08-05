from lib import *
from agent import Agent
import argparse

def test(env, agent, num_test_eps, results_basepath, render=True):
    """
        Hàm chơi thử game
        Parameters
        ---
        env: gym.Env
            môi trường để thử nghiệm
        agent: Agent
            Agent của trò chơi
        num_test_eps: int
            Số lần xem agent chơi thử
        results_basepath: str
            Đường dẫn lưu file kết quả
        render: bool
            Bật hoặc tắt render UI trò chơi.
        Returns
        ---
        none
    """

    step_cnt = 0
    # lịch sử phần thưởng
    reward_history = []

    # list kết quả để in vào file kết quả
    list_result = list()

    # vòng lặp quanh số lần chơi thử
    for ep in range(num_test_eps):
        score = 0
        done = False
        state = env.reset()
        while not done:

            if render:
                env.render()

            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            score += reward
            state = next_state
            step_cnt += 1

        reward_history.append(score)

        #in từng episode lên terminal
        print('Ep: {}, Score: {}\n'.format(ep, score))

        #Chuẩn bị và ghi file kết quả
        list_result.append('Ep: {}, Score: {}\n'.format(ep, score))
        with open(results_basepath+"result_reward.txt", "w") as f:
            for line in list_result:
                f.write(line)


if __name__ == "__main__":
    # parameter chọn số lần xem agent chơi thử
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_test_eps', type=int, default=100, help='số lần thực hiện chơi của agent')
    args = parser.parse_args()

    # Khởi tạo các tham số cần cho hàm test()
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=0.0, batch_size=64, n_actions=4, eps_end=0.00, input_dims=[8], lr=0.001)
    agent.load_model('{}/dqn_model'.format("../weights"))

    #tiến hành test (chơi thử)
    test(env=env, agent=agent, num_test_eps=args.num_test_eps, results_basepath="../run/")

    env.close()
