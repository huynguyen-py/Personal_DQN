
#### Reinforcement Q Learning - LunarLander-V2


Trong project này, mình cài đặt và ứng dụng mô hình Deep Q-Learning học chơi game LunarLander-V2. Kiến trúc mô hình là một mạng CNN đơn giản xây dựng trên module nn (Pytorch). Và sử dụng kỹ thuật Experience Replay cho quá trình chọn mẫu huẩn luyện. 

# Cấu hình thiết bị huấn luyện
```
Quá trình train trong 1 giờ đồng hồ với 377 episode
score 281.78 avarage score 119.22 epsilon 0.01

#####Cấu hình máy:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 471.41       Driver Version: 471.41       CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A    0C    P8    N/A /  N/A |     75MiB /  4096MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

You need to change the path in the variable "baseUrl" in the file "\promotion\promotion-ui\src\apis\jsonPlaceholder.js" to match the path to promotion-storage, in order to send the request to get the object" promotions".

```

# Tài liệu mô tả hoạt động
```
/document/doc/mota_hoatdong.docx
/document/doc/presentation.pptx
/document/doc/history_trained_one_hours.txt
```
# Clip demo trước, giữa và sau khi train
```
/demo/clipdemo.zip
```

# Nơi lưu trữ kết quả chơi thử của agent
```
/run/result_reward.txt
```

# Cài Đặt Môi Trường
Để cài đặt môi trường gõ lệnh sau
```
pip install -r requirements.txt
```


# Tiến hành huấn luyện và xem quá trình học gõ lệnh sau:
```
python /source/main.py
```

# Tiến hành xem agent chơi thử trò chơi gõ lệnh sau (num_test_eps đại diện cho số lần xem agent chơi):
```
python /source/agent_play_game.py --num_test_eps=10

```

# Liên hệ
Nếu bạn có bất kì vấn đề gì, vui lòng tạo issue hoặc liên hệ mình tại iamhuynguyen1002@gmail.com 
