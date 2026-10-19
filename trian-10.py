import os
import argparse

## Parse CLI arguments ##
def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument( '--scale_lr', action='store_true', help="Whether to scale lr.")
    parser.add_argument( '--report_to', type=str, default="wandb", choices=["wandb"],
                        help="Where to log to. Currently only supports wandb")
    parser.add_argument('--dataset', type=str, default='flare', required=True,
                        help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
    parser.add_argument( '--logging_dir', type=str, default="logs", help="Logging dir.")
    parser.add_argument( '--output_dir', type=str, default="output", help="Output dir.")
    parser.add_argument('--mixed_precision', type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to do mixed precision")
    parser.add_argument( '--img_folder', type=str, default='ISBI2016_ISIC_Part3B_Training_Data',
                        help='The image file path from data_path')
    parser.add_argument( '--csv_file', type=str, default='ISBI2016_ISIC_Part3B_Training_GroundTruth.csv',
                        help='The csv file to load in from data_path')
    parser.add_argument( '--self_condition', action='store_true', help='Whether to do self condition')
    parser.add_argument( '--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument( '--adam_beta1', type=float, default=0.95,
                        help='The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help='The beta2 parameter for the Adam optimizer.')
    parser.add_argument( '--adam_weight_decay', type=float, default=1e-6,
                        help='Weight decay magnitude for the Adam optimizer.')
    parser.add_argument( '--adam_epsilon', type=float, default=1e-08,
                        help='Epsilon value for the Adam optimizer.')
    parser.add_argument('--use_lion', type=bool, default=False, help='use Lion optimizer')
    parser.add_argument('--mask_channels', type=int, default=1, help='input channels for training (default: 3)')
    parser.add_argument('--input_img_channels', type=int, default=3,
                        help='output channels for training (default: 3)')
    parser.add_argument( '--image_size', type=int, default=128, help='input image size (default: 128)')
    parser.add_argument( '--data_path', default='./data', help='directory of input image')
    parser.add_argument( '--dim', type=int, default=64, help='dim (default: 64)')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs (default: 10000)')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps (default: 1000)')
    parser.add_argument('--save_every', type=int, default=100, help='save_every n epochs (default: 100)')
    parser.add_argument('--load_model_from', default=None, help='path to pt file to load from')
    parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
    parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
    parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
    parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
    parser.add_argument('--cache_rate', type=float, default=0.0, help='Cache rate to cache your dataset into GPUs')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--lstm_layers', type=int, default=4, help='lstm_layers')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden_size')
    parser.add_argument('--sampling_timesteps', type=int, default=250,help='number of sampling_timesteps (default: 250)')
    parser.add_argument('--local_rank', type=int, default=-1, help='Used for multi-process training')
    parser.add_argument('--steps_per', type=int, default=240,)
    parser.add_argument('--fft_hidden_size', type=int, default=4, help='fft_hidden_size')
    return parser.parse_args()

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
#from lion_pytorch import Lion
#from load_datasets_transforms4 import data_loader, data_transforms
from diff_unet76 import Unet, MedSegDiff
#from med_seg_diff_pytorch.dataset import ISICDataset, GenericNpyDataset
#from accelerate import Accelerator
#import matplotlib.pyplot as plt
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.data.image_reader import NibabelReader
from monai.transforms import (
         Lambdad
    )
from monai.transforms import LambdaD
from monai.metrics import DiceMetric
from monai.networks import one_hot  # 正确导入路径
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ResizeWithPadOrCrop,
    Lambda,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandRotate90d,
    RandCoarseDropoutd,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
)
import glob
import wandb
from torch.amp import GradScaler
import torch.nn.utils as nn_utils

device = torch.device("cuda:0")

if device.type == 'cuda':
    # 已分配给张量的显存
    allocated = torch.cuda.memory_allocated(device)
    # 已保留给缓存分配器但未实际分配给张量的显存
    reserved  = torch.cuda.memory_reserved(device)
    # 可释放缓存后可用的显存
    free = reserved - allocated
    # 总显存
    total = torch.cuda.get_device_properties(device).total_memory

    print(f"Total    : {total/1024**2:.1f} MB")
    print(f"Allocated: {allocated/1024**2:.1f} MB")
    print(f"Reserved : {reserved/1024**2:.1f} MB")
    print(f"Free     : {free/1024**2:.1f} MB")
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

free = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
print(f"可用显存：{free / 1024**2:.2f} MB")
# torch.autograd.set_detect_anomaly(True)
def load_data(args):
    # Load dataset
    out_classes = 7
    if args.dataset == 'LNCTVSeg':
        out_classes = 7


    if args.mode == 'train':
        train_samples = {}
        valid_samples = {}

        # 训练数据
        train_img = sorted(glob.glob(os.path.join(args.data_path, 'imagesTr', '*.nii.gz')))
        train_label = sorted(glob.glob(os.path.join(args.data_path, 'labelsTr', '*.nii.gz')))
        # 若需要重复标签（原代码中对 labels 进行了重复处理），可根据需求调整：
        train_label = [label for label in train_label for _ in range(2)]
        train_samples['images'] = train_img
        train_samples['labels'] = train_label

        # 验证数据
        valid_img = sorted(glob.glob(os.path.join(args.data_path, 'imagesTs', '*.nii.gz')))
        valid_label = sorted(glob.glob(os.path.join(args.data_path, 'labelsTs', '*.nii.gz')))
        valid_label = [label for label in valid_label for _ in range(2)]
        valid_samples['images'] = valid_img
        valid_samples['labels'] = valid_label

        print(f"len(train_img): {len(train_img)}, len(train_label): {len(train_label)}, "
              f"len(valid_img): {len(valid_img)}, len(valid_label): {len(valid_label)}")
        print('Finished loading training samples for dataset: {}'.format(args.dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))
        return train_samples, valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}
        test_img = sorted(glob.glob(os.path.join(args.data_path, 'imagesTs', '*.nii.gz')))
        test_samples['images'] = test_img
        print('Finished loading inference samples for dataset: {}'.format(args.dataset))
        return test_samples, out_classes

    # else:
    #     raise NotImplementedError("Mode {} is not supported.1".format(args.mode))

def data_transforms(args):
    """
    定义 MONAI 转换管道：
      - 训练模式下返回 (train_transforms, val_transforms)
      - 测试模式下返回 test_transforms
    其中，每个转换管道都将 nii.gz 数据转换为字典格式：
      { "image": Tensor, "label": Tensor } （训练/验证）或 { "image": Tensor } （测试）
    """


    if args.mode == 'train':
        crop_sample = args.crop_sample
        if args.dataset == 'LNCTVSeg':
            train_transforms = Compose([
                #LoadImaged(keys=["image", "label"],ensure_channel_first=True,reader=NibabelReader()),
                LoadImaged(keys=["image", "label"],  reader=NibabelReader(), image_only=False),
                # Lambdad(keys=["image", "label"], func=lambda x: (print(
                #     f"1.加载后尺寸: {x.shape}, 类型: {type(x)}"), x)[1]),

                # Step 2: 通道优先
                EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
                # Lambdad(keys=["image"], func=lambda x: print(
                #     f"2.通道调整后图像尺寸: {x.shape}")),

                # Step 3: 空间重采样
                Spacingd(keys=["image", "label"], pixdim=(1.2, 1.0, 1.0), mode=("bilinear", "nearest"),align_corners=False,padding_mode="border"),
                # Lambdad(keys=["image"], func=lambda x: (print(
                #     f"3.重采样后图像尺寸: {x.shape}, 强度范围: [{x.min():.2f}, {x.max():.2f}]"), x)[1]),
                Lambda(
                    func=lambda d: {
                        **d,
                        "image": ResizeWithPadOrCrop(
                            spatial_size=np.maximum(d["image"].shape[1:], d["label"].shape[1:]).tolist(),
                            mode="constant"
                        )(d["image"]),
                        "label": ResizeWithPadOrCrop(
                            spatial_size=np.maximum(d["image"].shape[1:], d["label"].shape[1:]).tolist(),
                            mode="constant"
                        )(d["label"]),
                    }
                ),
                # Step 4: 方向标准化
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Lambdad(keys=["image"], func=lambda x: (print(f"4.方向标准化后仿射矩阵:\n{x.affine}"), x)[1]),
                # Lambdad(keys=["image"], func=lambda x: (print(
                #     f"5.归一化前强度范围: [{x.min():.2f}, {x.max():.2f}]"), x)[1]),
                # Step 5: 强度归一化
                ScaleIntensityRanged(keys=["image"], a_min=-125, a_max=275, b_min=0.0, b_max=1.0, clip=True),
                # Lambdad(keys=["image"], func=lambda x:( print(
                #     f"5.归一化后强度范围: [{x.min():.2f}, {x.max():.2f}]"), x)[1]),
                # Lambdad(keys=["image"], func=lambda x: (print(
                #     f"剪裁前图像尺寸: {x.shape},"), x)[1]),
                # Lambdad(keys=["label"], func=lambda x: (print(
                #     f"剪裁前标签尺寸: {x.shape},"), x)[1]),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.image_size, args.image_size, args.image_size),
                    pos=1,
                    neg=1,
                    num_samples=args.crop_sample,
                    image_key="image",
                    image_threshold=0,
                ),
                # Lambdad(keys=["image"], func=lambda x: (print(
                #     f"剪裁前标签尺寸: {x.shape},"), x)[1]),
                # CropForegroundd(keys=["image", "label"], source_key="image"),
                # Lambdad(keys=["image"], func=lambda x: (print(
                #     f"剪裁后标签尺寸: {x.shape},"), x)[1]),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=0.8,
                    spatial_size=(args.image_size, args.image_size, args.image_size),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0, 1, 2],  # 允许X/Y/Z轴翻转
                    prob=0.5
                ),
                RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.1, 0.5),
                    sigma_y=(0.1, 0.5),
                    sigma_z=(0.1, 0.5),
                    prob=0.3
                ),
                RandAdjustContrastd(
                    keys=["image"],
                    gamma=(0.7, 1.3),  # 对比度调整范围
                    prob=0.3
                ),
                RandCoarseDropoutd(
                    keys=["image", "label"],
                    holes=10,  # 遮挡块数量
                    spatial_size=(10, 10, 10),  # 遮挡块大小
                    fill_value=0,  # 图像填充值
                    max_holes=20,
                    prob=0.4
                ),
                # LambdaD(keys=["label"], func=lambda x: (print(f"One-Hot前形状: {x.shape}"), x)[1]),
                AsDiscreted(keys="label", to_onehot=7),
                # LambdaD(keys=["label"], func=lambda x: (print(f"One-Hot后形状: {x.shape}"), x)[1]),
                ToTensord(keys=["image", "label"]),

            ])

            val_transforms = Compose([
                #LoadImaged(keys=["image", "label"],ensure_channel_first=True,reader=NibabelReader()),
                LoadImaged(keys=["image", "label"],  reader=NibabelReader(), image_only=False),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
                Spacingd(keys=["image", "label"], pixdim=(1.2, 1.0, 1.0),
                         mode=("bilinear", "nearest")),
                Lambda(
                    func=lambda d: {
                        **d,
                        "image": ResizeWithPadOrCrop(
                            spatial_size=np.maximum(d["image"].shape[1:], d["label"].shape[1:]).tolist(),
                            mode="constant"
                        )(d["image"]),
                        "label": ResizeWithPadOrCrop(
                            spatial_size=np.maximum(d["image"].shape[1:], d["label"].shape[1:]).tolist(),
                            mode="constant"
                        )(d["label"]),
                    }
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                AsDiscreted(keys="label", to_onehot=7),
                Lambdad(keys=["label"], func=lambda x: (print(f"One-Hot后形状: {x.shape}"), x)[1]),
                ToTensord(keys=["image", "label"]),

            ])
            print('Performed Data Augmentations for training samples!')
            return train_transforms, val_transforms
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented in transforms.")

    elif args.mode == 'test':
        if args.dataset == 'LNCTVSeg':
            test_transforms = Compose([
                LoadImaged(keys=["image", "label"],ensure_channel_first=True,reader=NibabelReader(), image_only=False),
                # EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.2), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ])
            return test_transforms
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented in transforms.")

    else:
        raise NotImplementedError("Mode {} is not supported.2".format(args.mode))


def main():
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_samples, valid_samples, out_classes = load_data(args)
    dice_val_best = 0.0
    train_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
    ]

    val_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
    ]

    train_transforms, val_transforms = data_transforms(args)
    print("检查第一个训练样本的结构:", train_files[0])
    ## Train Pytorch Data Loader and Caching
    print('Start caching datasets!')
    try:
        train_ds = CacheDataset(
            data=train_files, transform=train_transforms,
            cache_rate=args.cache_rate, num_workers=args.num_workers)
        test_sample = train_ds[0]
        print("\n训练样本调试信息:")
        print(type(test_sample))  # 应该输出<class 'dict'>
        # print(test_sample.keys())
        # print(f"1：Image shape: {test_sample['image'].shape}, Label shape: {test_sample['label'].shape}")
        # print(f"Image dtype: {test_sample['image'].dtype}, Label dtype: {test_sample['label'].dtype}")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
        batch = next(iter(train_loader))
        print(f"2：Image shape: {batch['image'].shape}, Label shape: {batch['label'].shape}")

    except Exception as e:
        print(f"加载训练数据失败: {str(e)}")
        raise
    ## Valid Pytorch Data Loader and Caching
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    test_sample = val_ds[0]
    # print("\n训练样本调试信息:")
    # print(f"3：Image shape: {test_sample['image'].shape}, Label shape: {test_sample['label'].shape}")
    # print(f"Image dtype: {test_sample['image'].dtype}, Label dtype: {test_sample['label'].dtype}")
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)
    batch = next(iter(val_loader))
    print(f"4：Image shape: {batch['image'].shape}, Label shape: {batch['label'].shape}")
    # 定义模型（已替换成新模型）
    print(f"加载训练数据完成！！可喜可贺")
    # dice_metric = DiceMetric(
    #     include_background=True,
    #     reduction="mean",
    #     get_not_nans=False
    # )
    model = Unet(
        dim=args.dim,
        image_size=args.image_size,
        mask_channels=args.mask_channels,
        input_img_channels=args.input_img_channels,
        resnet_block_groups=8,
        lstm_layers=args.lstm_layers,
        hidden_size=args.hidden_size,
        fft_hidden_size=args.fft_hidden_size,
        dynamic_ff_parser_attn_map=True,
        # skip_connect_condition_fmaps=True
        # self_condition=args.self_condition
    ).cuda()

    diffusion = MedSegDiff(model, timesteps=args.timesteps, sampling_timesteps=args.sampling_timesteps).cuda()
    if args.load_model_from is not None:
        state_dict = torch.load(args.load_model_from)
        diffusion.model.load_state_dict(state_dict['model_state_dict'])
        print(f'Loaded model from {args.load_model_from}')

    # 初始化优化器
    optimizer = AdamW(
        diffusion.model.parameters(),
        lr=args.learning_rate,
        # weight_decay=1e-6
        # 记得调参
        # betas=(args.adam_beta1, args.adam_beta2),
        # weight_decay=args.adam_weight_decay,
        # eps=args.adam_epsilon
    )
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    # 学习率策略
    best_dice = 0.0
    # 训练与验证循环

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        scaler = GradScaler()
        for batch in tqdm(train_loader):
            img = batch["image"].to(device)
            mask = batch["label"].to(device)
            # print(f"img type: {img.dtype}, mask type: {mask.dtype}")
            optimizer.zero_grad()
            # print(f"img:{img.shape}")
            with torch.amp.autocast(device_type="cuda"):
                loss = diffusion(mask, img, epoch, args.epochs)
            for name, param in diffusion.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN梯度出现在: {name}")
                        # 重置梯度避免污染
                        param.grad = torch.zeros_like(param.grad)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            # nn_utils.clip_grad_norm_(
            #     diffusion.model.parameters(),
            #     max_norm=1,
            #     norm_type=1.0,
            #     error_if_nonfinite=False
            # )
            # 👣 scale gradients
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            # total_loss += loss.item() * img.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Training Loss: {avg_loss:.4f}')

        # 每20个epoch验证并保存checkpoint
        # if (epoch + 1) % 1 == 0:
        #     model.eval()
        #     dice_metric.reset()
        #     val_loss = 0.0
        #     with torch.no_grad():
        #         for batch in tqdm(val_loader):
        #             img, label = batch["image"].cuda(), batch["label"].cuda()
        #             print(f"img:{img.shape}")
        #             print(f"label:{label.shape}")
        #             pred = diffusion.sample(img)
        #             print(f"pred type:{pred.dtype}")
        #             dice_metric(y_pred=pred, y=label)
        #
        #     avg_dice = dice_metric.aggregate().item()
        #     print(f'Validation Dice at epoch {epoch + 1}: {avg_dice:.4f}')
        #     if(avg_dice > dice_val_best):
        #         # 保存checkpoint
        #         checkpoint_path = os.path.join(
        #             checkpoint_dir,
        #             f'model_best_dice.pth'
        #         )
        #         torch.save({
        #             'epoch': epoch + 1,
        #             'model_state_dict': diffusion.model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'dice': avg_dice,
        #         }, checkpoint_path)
        #
        #         print(f"Checkpoint saved at {checkpoint_path} with Dice {avg_dice:.4f},and current best_dice is {dice_val_best:.4f}")
        #         dice_val_best = avg_dice
        if (epoch + 1) % 10 == 0:
            # 保存checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'model_dice{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'dice': avg_dice,
            }, checkpoint_path)

            print(f"Checkpoint saved at {checkpoint_path} ")

if __name__ == '__main__':
    main()
