import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
import wandb
import os
from utils.utils import prepare_compact_logger
from datasets.pcnhdf5dataset import PCNImageHDF5Dataset
from models.gan import TreeGANModule, TreeGANMuVarModule, PCNGANModule, PCNGANMuVarModule, TreeGANRawModule
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


module_dict = {
    "treegan": TreeGANModule,
    "treeganmuvar": TreeGANMuVarModule,
    "pcngan": PCNGANModule,
    "pcnganmuvar": PCNGANMuVarModule,
    "treeganraw": TreeGANRawModule
}


def dataLoaders(args):
    print("[+] Loading the data...")
    folder = args.folder
    json = args.json
    batch_size = args.batch_size

    # trainDataset = PCNImageDataset(
    #     folder, json, mode='train', b_tag=args.b_tag, img_height=args.size, img_width=args.size, img_count=args.img_count)
    # testDataset = PCNImageDataset(
    #     folder, json, mode='test', b_tag=args.b_tag, img_height=args.size, img_width=args.size, img_count=args.img_count)
    # valDataset = PCNImageDataset(
    #     folder, json, mode='val', b_tag=args.b_tag, img_height=args.size, img_width=args.size, img_count=args.img_count)

    trainDataset = PCNImageHDF5Dataset(folder, json, mode='train', b_tag=args.b_tag,
                                       img_height=args.size, img_width=args.size, img_count=args.img_count)
    testDataset = PCNImageHDF5Dataset(folder, json, mode='test', b_tag=args.b_tag,
                                      img_height=args.size, img_width=args.size, img_count=args.img_count)
    valDataset = PCNImageHDF5Dataset(folder, json, mode='val', b_tag=args.b_tag,
                                     img_height=args.size, img_width=args.size, img_count=args.img_count)

    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True,
                             drop_last=True, num_workers=4, prefetch_factor=8, persistent_workers=True, pin_memory=True)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=4, prefetch_factor=8, persistent_workers=True, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=False,
                           drop_last=True, num_workers=4, prefetch_factor=8, persistent_workers=True, pin_memory=True)
    return trainLoader, testLoader, valLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, default="../ShapeNet", help="Folder containing the data")
    parser.add_argument("--json", type=str, default="final.json",
                        help="JSON file containing the data")
    parser.add_argument("--b_tag", type=str, default="depth",
                        help="Tag for the B Image")
    parser.add_argument("--img_count", type=int, default=3, help="Image count")
    parser.add_argument("--module", type=str, default="treegan",
                        help="Module to use")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log dir")
    parser.add_argument("--exp", type=str, default="exp", help="Experiment")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--latent_dim", type=int,
                        default=256, help="Latent dimension")
    parser.add_argument("--num_dense", type=int,
                        default=8192, help="Number of dense")
    parser.add_argument("--gen_freq", type=int, default=1,
                        help="Generator Training frequency")
    parser.add_argument("--dis_freq", type=int, default=1,
                        help="Discriminator Training frequency")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to start")
    parser.add_argument("--scheduler", type=str,
                        default="step", help="Scheduler")
    parser.add_argument("--gamma", type=float, default=0.85, help="Gamma")
    parser.add_argument("--n_epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--decay_epoch", type=int,
                        default=10, help="Decay epoch")
    parser.add_argument("--save_iter", type=int,
                        default=20, help="Save interval")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--modelPath", type=str,
                        default="bestModel.pth", help="Path to model")
    parser.add_argument("--test", action="store_true", help="Test model")
    parser.add_argument("--testSave", action="store_true",
                        help="Save test output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training")
    parser.add_argument("--lambda_pixel", type=float,
                        default=10, help="pixelwise loss weight")
    parser.add_argument("--lambda_latent", type=float,
                        default=0.5, help="latent loss weight")
    parser.add_argument("--lambda_kl", type=float,
                        default=0.01, help="kullback-leibler loss weight")
    parser.add_argument("--lambda_chamfer", type=float,
                        default=1, help="chamfer loss weight")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_compact_logger(args.log_dir, args.exp)
    name = args.exp + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb_logger = WandbLogger(name=args.exp, project="TreeGAN",
                               log_model="all", save_dir=os.path.join(args.log_dir, args.exp))
    tensorboard_logger = SummaryWriter(
        log_dir=os.path.join(args.log_dir, args.exp, 'train')
    )
    for k, v in vars(args).items():
        wandb_logger.log_hyperparams({k: v})

    train_loader, test_loader, val_loader = dataLoaders(args)
    if args.module not in module_dict:
        print("Module not found")
        exit(1)
    else:
        model = module_dict[args.module](args)
    model.to(device)
    model.setup_ops(len(train_loader) * args.n_epochs)
    model.writer = tensorboard_logger
    if args.test:
        model.load_model(args.modelPath)
        model.eval()
        model.test(test_loader=train_loader)
    else:
        if args.resume:
            model.load_model(args.modelPath)
        model.train()
        # profiler = AdvancedProfiler(dirpath=os.path.join(
        #     args.log_dir, args.exp), filename='adv_profiler.txt')
        trainer = pl.Trainer(precision='16-mixed',
                             max_epochs=args.n_epochs,
                             logger=wandb_logger,
                             #  profiler=profiler,
                             )
        trainer.fit(model, train_loader, val_loader)
        wandb.finish()
