from recbole_custom.trainer import Trainer


class MMHPTrainer(Trainer):
    def __init__(self, config, model):
        super(MMHPTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        """
        重写训练 Epoch，加入 E-Step
        来源: HyperHawkes PDF page 4 (E-M 框架)
        """
        # --- E-Step: 意图推断 ---
        # 固定网络参数，更新聚类中心和概率分布
        self.model.eval()  # 某些层可能需要 eval 模式，但 E-Step 需要 no_grad
        self.model.e_step()

        # --- M-Step: 参数更新 ---
        # 正常的梯度下降训练
        self.model.train()
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)