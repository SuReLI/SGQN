import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import utils
import os
import kornia
import random

places_dataloader = None
places_iter = None


def _load_places(batch_size=256, image_size=84, num_workers=8, use_val=False):
    global places_dataloader, places_iter
    partition = "val" if use_val else "train"
    print(f"Loading {partition} partition of places365_standard...")
    for data_dir in utils.load_config("datasets"):
        if os.path.exists(data_dir):
            fp = os.path.join(data_dir, "places365_standard", partition)
            if not os.path.exists(fp):
                print(f"Warning: path {fp} does not exist, falling back to {data_dir}")
                fp = data_dir
            places_dataloader = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                    fp,
                    TF.Compose(
                        [
                            TF.RandomResizedCrop(image_size),
                            TF.RandomHorizontalFlip(),
                            TF.ToTensor(),
                        ]
                    ),
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            places_iter = iter(places_dataloader)
            break
    if places_iter is None:
        raise FileNotFoundError(
            "failed to find places365 data at any of the specified paths"
        )
    print("Loaded dataset from", data_dir)


def _get_places_batch(batch_size):
    global places_iter
    try:
        imgs, _ = next(places_iter)
        if imgs.size(0) < batch_size:
            places_iter = iter(places_dataloader)
            imgs, _ = next(places_iter)
    except StopIteration:
        places_iter = iter(places_dataloader)
        imgs, _ = next(places_iter)
    return imgs.cuda()


def random_overlay(x, dataset="places365_standard"):
    """Randomly overlay an image from Places"""
    global places_iter
    alpha = 0.5

    if dataset == "places365_standard":
        if places_dataloader is None:
            _load_places(batch_size=x.size(0), image_size=x.size(-1))
        imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1) // 3, 1, 1)
    else:
        raise NotImplementedError(
            f'overlay has not been implemented for dataset "{dataset}"'
        )

    return ((1 - alpha) * (x / 255.0) + (alpha) * imgs) * 255.0


def attribution_augmentation(x, mask, dataset="places365_standard"):
    """Complete non importnant pixels with a random image from Places"""
    global places_iter

    if dataset == "places365_standard":
        if places_dataloader is None:
            _load_places(batch_size=x.size(0), image_size=x.size(-1))
        imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1) // 3, 1, 1)
    else:
        raise NotImplementedError(
            f'overlay has not been implemented for dataset "{dataset}"'
        )

    # s_plus = random_conv(x) * mask
    s_plus = x * mask
    s_tilde = (((s_plus) / 255.0) + (imgs * (torch.ones_like(mask) - mask))) * 255.0
    s_minus = imgs * 255
    return s_tilde


def paired_aug(obs, mask):
    mask = mask.float()
    SEMANTIC = [kornia.augmentation.RandomAffine([-45., 45.], [0.3, 0.3], [0.5, 1.5], [0., 0.15]),kornia.augmentation.RandomErasing()]
    no_sem = lambda x : random_overlay(x)
    sem = random.sample(SEMANTIC,k=1)[0]
    img_out = no_sem(sem(obs))

    mask_out = sem(mask, sem._params)
    return img_out, mask_out

def attribution_random_patch_augmentation(
    x,
    cam,
    image_size=84,
    output_size=4,
    quantile=0.90,
    patch_proba=0.7,
    dataset="places365_standard",
):

    if dataset == "places365_standard":
        if places_dataloader is None:
            _load_places(batch_size=x.size(0), image_size=x.size(-1))
        negative = _get_places_batch(batch_size=x.size(0)).repeat(
            1, x.size(1) // 3, 1, 1
        )
    else:
        raise NotImplementedError(
            f'overlay has not been implemented for dataset "{dataset}"'
        )
    cam = cam.to(x.device)
    cam = F.adaptive_avg_pool2d(cam, output_size=output_size)
    q = torch.quantile(cam.flatten(1), quantile, 1)
    mask = (cam >= q[:, None, None]).long()
    exploration_mask = torch.rand(*mask.shape).to(x.device)
    exploration_mask[~mask] = 0
    expl_max = torch.amax(exploration_mask.view(mask.size(0), -1), dim=1)
    exploration_mask = (
        exploration_mask.view(-1, mask.size(1), mask.size(2)) == expl_max[:, None, None]
    ).long()
    bern = torch.bernoulli(torch.ones_like(mask) * patch_proba).long().to(x.device)
    selected_patch = (mask * bern) + exploration_mask
    selected_patch[selected_patch > 1] = 1
    selected_patch = F.upsample_nearest(selected_patch.float().unsqueeze(1), image_size)
    # augmented_x = (((0.5) * (x / 255.0) + (0.5) * negative) * 255.0) * selected_patch
    augmented_x = x * selected_patch
    complementary_mask = ~(selected_patch.bool())
    negative = negative * (complementary_mask.float())
    return augmented_x + (negative * 255)


def blending_augmentation(x, mask, overlay=True):
    s_plus, s_minus, s_tilde = attribution_augmentation(x, mask)
    if overlay:
        overlay_x = (1 - 0.5) * x + (0.5) * s_minus
        s_tilde = overlay_x * (~mask) + s_plus
    else:
        s_tilde = s_minus * (~mask) + s_plus
    return s_plus, s_minus, s_tilde


def random_conv(x):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.shape
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(x.device)
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        temp_x = F.pad(temp_x, pad=[1] * 4, mode="replicate")
        out = torch.sigmoid(F.conv2d(temp_x, weights)) * 255.0
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return total_out.reshape(n, c, h, w)


def batch_from_obs(obs, batch_size=32):
    """Copy a single observation along the batch dimension"""
    if isinstance(obs, torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.repeat(batch_size, 1, 1, 1)

    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, axis=0)
    return np.repeat(obs, repeats=batch_size, axis=0)


def prepare_pad_batch(obs, next_obs, action, batch_size=32):
    """Prepare batch for self-supervised policy adaptation at test-time"""
    batch_obs = batch_from_obs(torch.from_numpy(obs).cuda(), batch_size)
    batch_next_obs = batch_from_obs(torch.from_numpy(next_obs).cuda(), batch_size)
    batch_action = torch.from_numpy(action).cuda().unsqueeze(0).repeat(batch_size, 1)

    return random_crop_cuda(batch_obs), random_crop_cuda(batch_next_obs), batch_action


def identity(x):
    return x


def random_shift(imgs, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    return kornia.augmentation.RandomCrop((h, w))(imgs)


def random_crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (
        w1 is not None and h1 is not None
    ), "must either specify both w1 and h1 or neither of them"
    assert isinstance(x, torch.Tensor) and x.is_cuda, "input must be CUDA tensor"

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(
        x.shape
    ), "window_shape must be a tuple with same number of dimensions as x"

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3),
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)
