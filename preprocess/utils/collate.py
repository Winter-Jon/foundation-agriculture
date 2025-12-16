import numpy as np
import torch

from timm.data.mixup import Mixup, mixup_target, cutmix_bbox_and_lam


def fast_collate_with_key(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)
    
    Modified to automatically handle (key, image, target) tuples.
    """
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    
    # 自动检测是否包含 key
    # 假设结构为 (key, img, target) -> len=3
    # 假设结构为 (img, target)      -> len=2
    if len(batch[0]) == 3:
        has_keys = True
        key_idx, img_idx, tgt_idx = 0, 1, 2
    else:
        has_keys = False
        img_idx, tgt_idx = 0, 1
        
    if isinstance(batch[0][img_idx], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        is_np = isinstance(batch[0][img_idx][0], np.ndarray)
        inner_tuple_size = len(batch[0][img_idx])
        flattened_batch_size = batch_size * inner_tuple_size
        
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][img_idx][0].shape), dtype=torch.uint8)
        
        # 如果有 keys，我们需要初始化一个列表来存放对齐后的 keys
        keys = [None] * flattened_batch_size if has_keys else None
        
        for i in range(batch_size):
            assert len(batch[i][img_idx]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                # 计算展平后的索引位置
                flat_idx = i + j * batch_size
                
                targets[flat_idx] = batch[i][tgt_idx]
                
                if has_keys:
                    keys[flat_idx] = batch[i][key_idx]
                    
                if is_np:
                    tensor[flat_idx] += torch.from_numpy(batch[i][img_idx][j])
                else:
                    tensor[flat_idx] += batch[i][img_idx][j]
                    
        return (keys, tensor, targets) if has_keys else (tensor, targets)

    elif isinstance(batch[0][img_idx], np.ndarray):
        targets = torch.tensor([b[tgt_idx] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][img_idx].shape), dtype=torch.uint8)
        
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][img_idx])
            
        if has_keys:
            keys = [b[key_idx] for b in batch]
            return keys, tensor, targets
        return tensor, targets

    elif isinstance(batch[0][img_idx], torch.Tensor):
        targets = torch.tensor([b[tgt_idx] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][img_idx].shape), dtype=torch.uint8)
        
        for i in range(batch_size):
            tensor[i].copy_(batch[i][img_idx])
            
        if has_keys:
            keys = [b[key_idx] for b in batch]
            return keys, tensor, targets
        return tensor, targets

    else:
        assert False, f"Unknown type for image data: {type(batch[0][img_idx])}"


class FastCollateMixupWithKey(Mixup):
    """ Fast Collate w/ Mixup/Cutmix that applies different params to each element or whole batch
    
    Modified to handle batch structure: (key, image, target)
    """

    def _mix_elem_collate(self, output, batch, half=False):
        batch_size = len(batch)
        num_elem = batch_size // 2 if half else batch_size
        assert len(output) == num_elem
        lam_batch, use_cutmix = self._params_per_elem(num_elem)
        # 【修改点】图片在 index 1
        is_np = isinstance(batch[0][1], np.ndarray)

        for i in range(num_elem):
            j = batch_size - i - 1
            lam = lam_batch[i]
            # 【修改点】取图片用 index 1
            mixed = batch[i][1] 
            if lam != 1.:
                if use_cutmix[i]:
                    if not half:
                        mixed = mixed.copy() if is_np else mixed.clone()
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam,
                    )
                    # 【修改点】取图片用 index 1
                    mixed[:, yl:yh, xl:xh] = batch[j][1][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    if is_np:
                        # 【修改点】取图片用 index 1
                        mixed = mixed.astype(np.float32) * lam + batch[j][1].astype(np.float32) * (1 - lam)
                        np.rint(mixed, out=mixed)
                    else:
                        # 【修改点】取图片用 index 1
                        mixed = mixed.float() * lam + batch[j][1].float() * (1 - lam)
                        torch.round(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8)) if is_np else mixed.byte()
        if half:
            lam_batch = np.concatenate((lam_batch, np.ones(num_elem)))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_pair_collate(self, output, batch):
        batch_size = len(batch)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        # 【修改点】图片在 index 1
        is_np = isinstance(batch[0][1], np.ndarray)

        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            # 【修改点】取图片用 index 1
            mixed_i = batch[i][1]
            mixed_j = batch[j][1]
            assert 0 <= lam <= 1.0
            if lam < 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam,
                    )
                    patch_i = mixed_i[:, yl:yh, xl:xh].copy() if is_np else mixed_i[:, yl:yh, xl:xh].clone()
                    mixed_i[:, yl:yh, xl:xh] = mixed_j[:, yl:yh, xl:xh]
                    mixed_j[:, yl:yh, xl:xh] = patch_i
                    lam_batch[i] = lam
                else:
                    if is_np:
                        mixed_temp = mixed_i.astype(np.float32) * lam + mixed_j.astype(np.float32) * (1 - lam)
                        mixed_j = mixed_j.astype(np.float32) * lam + mixed_i.astype(np.float32) * (1 - lam)
                        mixed_i = mixed_temp
                        np.rint(mixed_j, out=mixed_j)
                        np.rint(mixed_i, out=mixed_i)
                    else:
                        mixed_temp = mixed_i.float() * lam + mixed_j.float() * (1 - lam)
                        mixed_j = mixed_j.float() * lam + mixed_i.float() * (1 - lam)
                        mixed_i = mixed_temp
                        torch.round(mixed_j, out=mixed_j)
                        torch.round(mixed_i, out=mixed_i)
            output[i] += torch.from_numpy(mixed_i.astype(np.uint8)) if is_np else mixed_i.byte()
            output[j] += torch.from_numpy(mixed_j.astype(np.uint8)) if is_np else mixed_j.byte()
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_batch_collate(self, output, batch):
        batch_size = len(batch)
        lam, use_cutmix = self._params_per_batch()
        # 【修改点】图片在 index 1
        is_np = isinstance(batch[0][1], np.ndarray)

        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                output.shape,
                lam,
                ratio_minmax=self.cutmix_minmax,
                correct_lam=self.correct_lam,
            )
        for i in range(batch_size):
            j = batch_size - i - 1
            # 【修改点】取图片用 index 1
            mixed = batch[i][1]
            if lam != 1.:
                if use_cutmix:
                    mixed = mixed.copy() if is_np else mixed.clone()
                    # 【修改点】取图片用 index 1
                    mixed[:, yl:yh, xl:xh] = batch[j][1][:, yl:yh, xl:xh]
                else:
                    if is_np:
                        # 【修改点】取图片用 index 1
                        mixed = mixed.astype(np.float32) * lam + batch[j][1].astype(np.float32) * (1 - lam)
                        np.rint(mixed, out=mixed)
                    else:
                        # 【修改点】取图片用 index 1
                        mixed = mixed.float() * lam + batch[j][1].float() * (1 - lam)
                        torch.round(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8)) if is_np else mixed.byte()
        return lam

    def __call__(self, batch, _=None):
        batch_size = len(batch)
        assert batch_size % 2 == 0, 'Batch size should be even when using this'
        
        # 【修改点】提取 Keys (index 0)
        keys = [b[0] for b in batch]
        
        half = 'half' in self.mode
        if half:
            batch_size //= 2
            # 【修改点】如果是 half 模式，keys 也要切片
            keys = keys[:batch_size]
            
        # 【修改点】初始化 output 形状时，图片在 index 1
        output = torch.zeros((batch_size, *batch[0][1].shape), dtype=torch.uint8)
        
        if self.mode == 'elem' or self.mode == 'half':
            lam = self._mix_elem_collate(output, batch, half=half)
        elif self.mode == 'pair':
            lam = self._mix_pair_collate(output, batch)
        else:
            lam = self._mix_batch_collate(output, batch)
            
        # 【修改点】Target 在 index 2
        target = torch.tensor([b[2] for b in batch], dtype=torch.int64)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        target = target[:batch_size]
        
        # 【修改点】返回三元组 (keys, img, target)
        return keys, output, target