import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Dict, List

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NatureCNN(BaseFeaturesExtractor):
    """(수정 없음) 이미지 입력을 처리하는 CNN 부분은 그대로 둡니다."""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # PyTorch는 채널 우선(channel-first)이므로, 입력 채널 수를 observation_space.shape[0]으로 설정해야 합니다.
        # 환경의 'screens' 데이터가 (높이, 너비, 채널) 순서이므로 변환이 필요합니다.
        # 여기서는 입력 데이터가 이미 (채널, 높이, 너비)로 온다고 가정합니다.
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 입력 텐서의 차원을 (배치, 채널, 높이, 너비) 순서로 변경합니다.
        return self.linear(self.cnn(observations))

class CombinedExtractor(BaseFeaturesExtractor):
    """
    (수정됨) RedGymEnvAgentic의 Dict 관측 공간을 처리하도록 수정된 특징 추출기.
    """
    def __init__(self, observation_space: spaces.Dict):
        # --- 1. 각 부분의 크기 계산 ---
        # 이미지(screens) 특징 추출기
        cnn_output_dim = 512
        
        # 상태 벡터(나머지) 특징 추출기
        mlp_output_dim = 128
        
        # 나머지 관측 값들의 전체 크기 계산
        # 'screens', 'map' 등 이미지 형태의 데이터는 제외합니다.
        state_vector_size = 0
        self.state_keys = []
        for key, subspace in observation_space.spaces.items():
            if key != "screens" and isinstance(subspace, spaces.Box) or isinstance(subspace, spaces.MultiBinary):
                # .shape을 사용하여 다차원 데이터도 처리
                state_vector_size += int(torch.prod(torch.tensor(subspace.shape)))
                self.state_keys.append(key)
        
        # 최종 특징 벡터의 총 차원
        total_features_dim = cnn_output_dim + mlp_output_dim
        
        super().__init__(observation_space, features_dim=total_features_dim)

        # --- 2. 각 추출기 정의 ---
        self.cnn = NatureCNN(observation_space["screens"], features_dim=cnn_output_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(state_vector_size, 256),
            nn.ReLU(),
            nn.Linear(256, mlp_output_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # --- 3. 데이터 처리 및 결합 ---
        # 이미지(screens) 데이터 처리
        cnn_features = self.cnn(observations["screens"])
        
        # 나머지 상태 데이터들을 하나의 벡터로 결합
        state_tensors: List[torch.Tensor] = []
        for key in self.state_keys:
            # .float()으로 타입을 통일하고, .flatten(start_dim=1)으로 1차원으로 만듭니다.
            state_tensors.append(observations[key].flatten(start_dim=1).float())
        
        state_vector = torch.cat(state_tensors, dim=1)
        
        # 결합된 상태 벡터를 MLP로 처리
        mlp_features = self.mlp(state_vector)
        
        # 최종적으로 CNN 특징과 MLP 특징을 결합하여 반환
        return torch.cat([cnn_features, mlp_features], dim=1)