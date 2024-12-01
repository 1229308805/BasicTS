import torch
import torch.nn as nn
from .glaff import Plugin

class TimeGLAFF(nn.Module):
    """
    只使用时间特征进行预测的模型，基于GLAFF架构
    """
    def __init__(self, **model_args):
        super(TimeGLAFF, self).__init__()
        self.hist_len = model_args["hist_len"]
        self.pred_len = model_args["pred_len"]
        self.time_of_day_size = model_args["time_of_day_size"]
        
        # 初始化GLAFF插件
        self.plugin = Plugin(**model_args)
        
    def _day_of_year_to_month_day_tensor(self, day_of_year_tensor, is_leap_year=False):
        # 定义每月的天数
        days_in_month = torch.tensor(
            [31, 29 if is_leap_year else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        )
        cumulative_days = torch.cumsum(days_in_month, dim=0)
        
        month_tensor = torch.zeros_like(day_of_year_tensor)
        day_tensor = torch.zeros_like(day_of_year_tensor)
        
        for month in range(12):
            if month == 0:
                mask = (day_of_year_tensor <= cumulative_days[month])
            else:
                mask = (day_of_year_tensor > cumulative_days[month - 1]) & (day_of_year_tensor <= cumulative_days[month])
            
            month_tensor[mask] = month + 1
            day_tensor[mask] = day_of_year_tensor[mask] - (cumulative_days[month - 1] if month > 0 else 0)
        
        return month_tensor, day_tensor

    def timeslot_to_time_tensor(self, time_of_day_tensor, time_of_day_size):
        total_seconds_per_day = 86400
        seconds_per_timeslot = total_seconds_per_day // time_of_day_size

        total_seconds_tensor = time_of_day_tensor * seconds_per_timeslot

        hours_tensor = total_seconds_tensor // 3600
        minutes_tensor = (total_seconds_tensor % 3600) // 60
        seconds_tensor = total_seconds_tensor % 60

        return hours_tensor, minutes_tensor, seconds_tensor

    def get_time_feature(self, time_features):
        time_features = time_features[:, :, 0, :]

        # time of day to seconds, minutes, hours
        time_of_day = time_features[..., 0] * self.time_of_day_size
        hours, minuts, seconds = self.timeslot_to_time_tensor(time_of_day, self.time_of_day_size)
        
        # day of year to month of year, day of month, day of week
        day_of_year = time_features[..., 3] * 366
        month, day = self._day_of_year_to_month_day_tensor(day_of_year)
        weekday = time_features[..., 1] * 7

        # 归一化时间特征
        month = month / 12 - 0.5
        day = day / 31 - 0.5
        weekday = weekday / 6 - 0.5
        hours = hours / 23 - 0.5
        minuts = minuts / 59 - 0.5
        seconds = seconds / 59 - 0.5

        time = torch.stack([month, day, weekday, hours, minuts, seconds], dim=-1).clone()
        return time

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        """
        Args:
            history_data (torch.Tensor): [B, L, N, C]
            future_data (torch.Tensor): [B, L, N, C]
        """

        # 提取时间特征
        x_time = self.get_time_feature(history_data[..., 1:]).clone()
        y_time = self.get_time_feature(future_data[..., 1:]).clone()
        x_data = history_data[..., 0].clone()
        y_data = future_data[..., 0].clone()
        
        # 使用GLAFF插件进行预测
        pred, reco, map2 = self.plugin(x_data, x_time, x_data[:, -self.pred_len:, :], y_time[:, -self.pred_len:, :])
        
        # 构造插件损失所需的张量
        plugin_prediction = torch.cat([pred, reco, pred, map2], dim=1)
        plugin_target = torch.cat([y_data, x_data, y_data, y_data], dim=1)
        #打印输出形状

        return {
            "prediction": pred.unsqueeze(-1),  # [B, L, N, 1]
            "plugin_prediction": plugin_prediction,
            "plugin_target": plugin_target
        } 
