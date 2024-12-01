#只使用时间戳进行预测，看看能到达怎么个精度
import torch
import torch.nn as nn
from argparse import Namespace

class TimeStamp(nn.Module):
    '''
    一个只使用时间戳特征进行预测的基准模型
    '''
    def __init__(self, **model_args):
        super(TimeStamp, self).__init__()
        configs = Namespace(**model_args)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.time_of_day_size = configs.time_of_day_size
        
        
        # 预测层
        self.projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, 1)
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """
        Args:
            history_data (Tensor): 输入数据，形状: [B, L1, N, C]
            future_data (Tensor): 未来数据，形状: [B, L2, N, C]

        Returns:
            torch.Tensor: 输出预测，形状 [B, L2, N, 1]
        """
        B, L2, N, _ = history_data.shape
        
        # 只使用时间特征进行预测
        time_of_day = history_data[..., 0,1:]  # 过去的时间戳特征  ['hour of day', 'day of week', 'month of year', 'holiday']
        #没有future_data输入，从历史数据中推导出未来的时间戳特征,从最后一个时间戳开始，依次推导出未来的时间戳特征
        future_time_of_day = []
        for i in range(self.pred_len):
            future_time_of_day.append(time_of_day[..., i:i+1])
        future_time_of_day = torch.cat(future_time_of_day, dim=-1)

        
        
        #
        # 编码时间特征
        time_embed = self.time_embedding(time_of_day)
        
        # 预测
        prediction = self.projection(time_embed)
        
        return prediction.unsqueeze(-1)  # [B, L2, N, 1]
