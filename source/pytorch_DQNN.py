from lib import *


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        """
                Hàm xây dựng mạng
                Parameters
                ---
                lr: float
                    siêu tham số tốc độ học tập
                input_dims: [int]
                    input dim fully connected 1
                fc1_dims: int
                    output dim fully connected 1 và  input dim fully connected 2
                fc2_dims: int
                    output dim fully connected 2 và  input dim fully connected 3
                n_actions: int
                    Số lượng action của agent trong môi trường.
                    Còn là output dim fully connected 3
                Returns
                ---
                none
        """
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
                Hàm xây lan truyền tuyến, khi implement module NN của pytorch cần overide hàm này
                Sẽ tự động gọi hàm này
                Ở lớp cuối cùng không sử dụng một activate function vì output có thể là số âm
                nên ưu tiên sử dụng dữ liệu thô
                Parameters
                ---
                state: array
                    state của agent
                Returns
                ---
                actions: [int] lenght = 4
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

