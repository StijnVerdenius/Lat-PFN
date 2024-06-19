import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, decay=1
    ):
        super().__init__(
            optimizer,
            T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )
        self.decay = decay
        self.initial_lrs = self.base_lrs

    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch)
                )
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
            else:
                n = 0

            self.base_lrs = [
                initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs
            ]

        super().step(epoch)
