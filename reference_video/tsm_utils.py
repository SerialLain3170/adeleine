# codes are from
# https://github.com/Pika7ma/Temporal-Shift-Module/blob/master/tsm_util.py
import torch
import torch.nn.functional as F


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, tensor):
        # not support higher order gradient
        # tensor = tensor.detach_()
        n, t, c, h, w = tensor.size()
        fold = c // 4
        ctx.fold_ = fold
        buffer_ = tensor.data.new(n, t, fold, h, w).zero_()
        buffer_[:, :-1] = tensor.data[:, 1:, :fold]
        tensor.data[:, :, :fold] = buffer_
        buffer_.zero_()
        buffer_[:, 1:] = tensor.data[:, :-1, fold: 2 * fold]
        tensor.data[:, :, fold: 2 * fold] = buffer_
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer_ = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer_[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer_
        buffer_.zero_()
        buffer_[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer_
        return grad_output, None


def tsm(tensor, version='zero', inplace=True):
    shape = B, T, C, H, W = tensor.shape
    split_size = C // 4
    if not inplace:
        pre_tensor, post_tensor, peri_tensor = tensor.split(
            [split_size, split_size, C - 2 * split_size],
            dim=2
        )
        if version == 'zero':
            pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]  # NOQA
            post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]  # NOQA
        elif version == 'circulant':
            pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],  # NOQA
                                     pre_tensor [:,   :-1, ...]), dim=1)  # NOQA
            post_tensor = torch.cat((post_tensor[:,  1:  , ...],  # NOQA
                                     post_tensor[:,   :1 , ...]), dim=1)  # NOQA
        else:
            raise ValueError('Unknown TSM version: {}'.format(version))
        return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(shape)
    else:
        out = InplaceShift.apply(tensor)
        return out