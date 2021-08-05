import torch

from lib import *
from pytorch_DQNN import DeepQNetwork

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000,
                 eps_end=0.01, eps_dec=5e-4):
        # Hệ số discount factor
        self.gamma = gamma
        # Siêu tham số epsilon tham lam
        self.epsilon = epsilon
        # Giá trị nhỏ nhất của epsilon
        self.eps_min = eps_end
        # Các bước giảm epsilon theo một khoảng 5e-4
        self.eps_dec = eps_dec
        # Siêu tham số learning rate của học sâu
        self.lr = lr
        # Nơi đây sẽ là tập học 4 action được chọn trong quá trình khám phá ngẫu nhiên
        self.action_space = [i for i in range(n_actions)]
        # Kích thước bộ nhớ
        self.mem_size = max_mem_size
        # Kích thước batch mỗi data đưa vào training
        self.batch_size = batch_size
        # Bộ đếm vị trí batch
        self.mem_cntr = 0
        #Mạng DQN cho quá trình học tổng quát Q-Learning
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        # Các bộ nhớ sử dụng cho data train gồm state, new_state, action, reward, terminal
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        """
                Hàm lưu trữ tiến trình chuyển tiếp, một mẫu
                Parameters
                ---
                state: gym.Env
                    state hiện tại

                action: int
                    hành động của action

                reward: float
                    phần thưởng

                state_: array
                    state mới

                done: bool
                    cờ hiệu kết thúc episode.
                Returns
                ---
                none
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        """
                Hàm chọn action cho agent. random chọn một số float
                Nhằm thực hiện cho quá trình chọn ngẫu nhiên một hành động
                ngẫu nhiên không tuần theo kĩ thuật Epsilon tham lam
                Parameters
                ---
                observation: array [8]
                    Quan sát của agent
                Returns
                ---
                action: int
                    Hành động thực hiện của agent
        """
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        """
                Hàm học tập của Agent thực hiện các kĩ thuật forward/backward của DQN và "experience replay"
                Parameters
                ---
                Returns
                ---
                none
        """

        # Ý nghĩa: nếu bộ nhớ experience replay chưa đủ thì không làm gì cả
        if self.mem_cntr < self.batch_size:
            return
        #  chúng ta cần đặt các gradient thành 0 trước khi bắt đầu thực hiện backpropragation
        #  vì PyTorch tích lũy các gradient trong các lần đi lùi tiếp theo
        self.Q_eval.optimizer.zero_grad()

        # Chọn batch và batch size data dùng để cho vào quá trình huấn luyện của mạng Fully Connected Neural
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Lấy giá trị state, new_state, reward, terminal theo batch đã chọn đưa vào device GPU hoặc CPU
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        # Chọn action batch
        action_batch = self.action_memory[batch]
        # print("index:",batch_index)
        # print("action:",action_batch)

        #Mạng eval chỉ dùng 64 giá trị argmax action đc chọn bởi 4*64 action
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        #Mạng next lấy tất cả 4*64 giá trị
        q_next = self.Q_eval.forward(new_state_batch)
        # print("eval", q_eval)
        # print("next", q_next)

        q_next[terminal_batch] = 0.0

        # tính target (mục tiêu) của action tại state
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        # Tính độ mất mát, để cập nhật lại khi backward() và optimizer
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # Cập nhật lại epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self, filename):
        """
                Hàm lưu model
                Parameters
                ---
                filename: string
                    tên file model
                ---
                none
        """
        torch.save(self.Q_eval.state_dict(), filename)

    def load_model(self, filename):
        """
                Hàm load state dict model
                Parameters
                ---
                filename: string
                    tên file model
                ---
                none
        """
        self.Q_eval.load_state_dict(torch.load(filename))
        self.Q_eval.eval()