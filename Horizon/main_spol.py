import argparse,datetime,time,json,os,random
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,Subset
from sklearn.metrics import roc_auc_score
# from timm.models import create_model
from network.spol_backbone import Net
from network.spol import SPOL_pretrain
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from misc import NativeScalerWithGradNormCount as NativeScaler
from engine.engine_spol import train_one_epoch,evaluate_cls, evaluate_dice
import utils,misc
from dataloader.dataset_mri import Brats2021Dataset
from dataloader.fully_datasets import insDataSet
from dataloader.datasets import build_dataset
from sklearn.metrics import roc_auc_score, roc_curve

os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.backends.cudnn.enabled=False

def get_args_parser():
    parser = argparse.ArgumentParser('Start', add_help=False)
    parser.add_argument('--batch-size', default=48, type=int)  # parser自动将'-'转化为'_'
    parser.add_argument('--epochs', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='build_model', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=240, type=int, help='images input size')

    # Optimizer parameters  给定参数创建优化器
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters  学习率调整策略
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',  # 30
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=2, metavar='N',  # 5
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')  # 10
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--lesion', default='Hemorrhage', type=str, help='Name of model to train')
    parser.add_argument('--data-path', default='D:/code/Proto_ICH/Datasets/pngs', type=str, help='dataset path')  # VOC12目录
    parser.add_argument('--img-list', default='D:/code/Proto_ICH/Datasets/rsna', type=str, help='image list path')  # img_list所在目录
    parser.add_argument('--data-set', default='RSNA', type=str, help='dataset')

    ###########################可修改########################
    parser.add_argument('--flair_ncct_mapping', action='store_true',
                        help='Apply the precomputed FLAIR-to-NCCT quantile map to Tumor data')
    parser.add_argument('--mapping_path', default='mia_revision/flair_ncct_mapping.json',
                        help='Existing fixed quantile mapping JSON; never fitted during training')
    parser.add_argument('--output_dir', default='mia_revision/spol_multi_bank_poit',
                        help='Use a separate output directory for each experiment')
    parser.add_argument('--spol-components', type=int, nargs='+', default=[1, 3, 5, 7, 10])
    parser.add_argument('--spol-bank-sizes', type=int, nargs='+', default=[500, 1000, 2000])
    parser.add_argument('--spol-component-bank-size', type=int, default=1000)
    parser.add_argument('--spol-capacity-components', type=int, default=3)
    parser.add_argument('--spol-cluster-bank-size', type=int, default=1000)
    parser.add_argument('--spol-thresholds', type=float, nargs='+', default=[0.8])
    parser.add_argument('--spol-num-prototypes', type=int, default=50)
    parser.add_argument('--spol-cluster-prototypes', type=int, default=50)
    parser.add_argument('--spol-gmm-init-iterations', '--spol-gmm-iterations',
                        dest='spol_gmm_init_iterations', type=int, default=50,
                        help='Maximum EM iterations for every fresh bank fit')
    parser.add_argument('--spol-gmm-covariance', choices=['diag', 'full'], default='full')
    parser.add_argument('--spol-gmm-covariance-shrinkage', type=float, default=0.25,
                        help='Full-covariance shrinkage toward its diagonal in [0, 1]')
    parser.add_argument('--spol-gmm-gamma',
                        dest='spol_gmm_gamma', type=float, default=0.90,
                        help='History retention for matched-EM sufficient statistics')
    parser.set_defaults(spol_gmm_update_rule='matched_em')
    parser.add_argument('--save-first-fit', action='store_true', help='Save the complete model once all GMM and OT banks have fitted both classes')
    parser.add_argument('--spol-no-cluster', action='store_true')
    parser.add_argument('--spol-bank-stat-epoch', type=int, default=10)
    parser.add_argument('--spol-bank-stat-rounds', type=int, default=2)
    #######################################################

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=2, type=int)  # 并行数
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--seed', default=42, type=int)

    return parser


def same_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def main(args):
    print(args);device=torch.device(args.device);
    same_seeds(args.seed);
    cudnn.benchmark=True
    if args.lesion=='Hemorrhage':
        args.output_dir=args.output_dir+'/hemorrhage/';Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        dataset_train,args.nb_classes=build_dataset(is_train=True, args=args)
        data_loader_train=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=args.pin_mem,drop_last=True)
        ins_val=insDataSet(base_dir=r"D:/code/Proto_ICH/Datasets",data_ls="/val_slices.txt")
        dice_loader=DataLoader(ins_val, batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=args.pin_mem)
        rsna_val,_=build_dataset(is_train=False,args=args)
        cls_loader=DataLoader(rsna_val, batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=args.pin_mem)
    else:
        args.nb_classes=1;args.output_dir=args.output_dir+'/tumor/';Path(args.output_dir).mkdir(parents=True,exist_ok=True)
        train_data=open(r"C:\Users\Caozhi\Desktop\prototype_weakly\data\train_data_2D.txt").read().splitlines()
        test_data=open(r"C:\Users\Caozhi\Desktop\prototype_weakly\data\test_data_2D.txt").read().splitlines()
        dataset_train=Brats2021Dataset('D:/Datasets/BRATS2021_Training_none_npy/', train_data, [240,240], stage='train', task='cls')
        dataset_seg=Brats2021Dataset('D:/Datasets/BRATS2021_Training_none_npy/', test_data, [240,240], stage='val', task='seg')
        dataset_cls = Brats2021Dataset('D:/Datasets/BRATS2021_Training_none_npy/', test_data, [240, 240], stage='val', task='cls')
        if args.flair_ncct_mapping:
            from flair_ncct_mapping import map_tumor_datasets
            dataset_train, dataset_seg, dataset_cls = map_tumor_datasets(
                (dataset_train, dataset_seg, dataset_cls), args.mapping_path)
        data_loader_train=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,drop_last=True)
        dice_loader=DataLoader(dataset_seg,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True, drop_last=True)
        cls_loader=DataLoader(dataset_cls, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)


    #########################可修改######################

    tumor_bank = (0.8, 1000, 5) if args.lesion != 'Hemorrhage' else None
    model = SPOL_pretrain(
        num_cls=1, num_prototype=args.spol_num_prototypes,
        num_components=args.spol_components, bank_sizes=args.spol_bank_sizes,
        thresholds=args.spol_thresholds, train_cluster=not args.spol_no_cluster,
        component_bank_size=args.spol_component_bank_size,
        capacity_components=args.spol_capacity_components,
        cluster_bank_size=args.spol_cluster_bank_size,
        cluster_num_prototypes=args.spol_cluster_prototypes,
        gmm_init_iterations=args.spol_gmm_init_iterations,
        gmm_covariance_type=args.spol_gmm_covariance,
        gmm_momentum_gamma=args.spol_gmm_gamma,
        gmm_covariance_shrinkage=args.spol_gmm_covariance_shrinkage,
        single_bank_config=tumor_bank,
    ).to(device)
    print("GMM banks:", {name: (model.bank_groups[name], config) for name, config in model.bank_configs.items()})
    print("OT banks:", model.cluster_bank_configs)
    model.progress_log_path = Path(args.output_dir) / "train_progress.jsonl"
    model.first_fit_checkpoint_path = Path(args.output_dir) / 'checkpoint-all-banks-first-fit.pth' if args.save_first_fit else None
    model.first_fit_checkpoint_args = args
    model.gmm_report_path = Path(args.output_dir) / "gmm_epoch_report.jsonl"
    if args.lesion != 'Hemorrhage':
        model.prototype_update_during_training = False
        model.bank_collection_epoch = args.spol_bank_stat_epoch
        model.bank_collection_rounds = args.spol_bank_stat_rounds
        model.bank_checkpoint_path = Path(args.output_dir) / 'checkpoint-bank.pth'
        model.bank_checkpoint_args = args
    model.progress_log_path.parent.mkdir(parents=True, exist_ok=True)

    ######################################################
    optimizer=create_optimizer(args,model); loss_scaler=torch.cuda.amp.GradScaler()
    misc.load_model(args=args, model_without_ddp=model, opt=optimizer, loss_scaler=loss_scaler)
    lr_scheduler,_=create_scheduler(args,optimizer)
    output_dir=Path(args.output_dir);
    n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)
    with (output_dir/'log.txt').open('a') as f:f.write(str(args)+'\n')

    best_acc = best_dice = 0
    best_threshold = 0.5

    for epoch in range(args.start_epoch, args.epochs):
        cls_stats = {}
        train_stats = {}
        dice = 0

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler)
        lr_scheduler.step(epoch)
        misc.save_model(args=args, model=model, opt=optimizer, loss_scaler=loss_scaler, epoch="last")
        #########################可修改######################
        if ((args.lesion == 'Hemorrhage' and epoch >= 10 and epoch % 2 == 0)
                or (args.lesion != 'Hemorrhage' and epoch % 2 == 0)):
        ####################################################
            if epoch == 10:
                misc.save_model(args=args, model=model, opt=optimizer, loss_scaler=loss_scaler, epoch="10")

            dice = evaluate_dice(model, dice_loader, device)
            cls_stats = evaluate_cls(model, cls_loader, device)
            acc = cls_stats["acc"]

            print(f"epoch:{epoch},acc:{acc:.4f},dice:{dice:.4f}")

            if acc > best_acc:
                best_acc = acc
                misc.save_model(args=args, model=model, opt=optimizer, loss_scaler=loss_scaler, epoch="best_acc")

            if dice > best_dice:
                best_dice = dice
                misc.save_model(args=args, model=model, opt=optimizer, loss_scaler=loss_scaler, epoch="best_dice")

        log = {**{f"train_{k}": v for k, v in train_stats.items()},
               **{f"cls_{k}": v for k, v in cls_stats.items()},
               "dice": dice,
               "epoch": epoch}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log) + "\n")

    print(f"best acc:{best_acc:.4f}, best dice:{best_dice:.4f}, threshold:{best_threshold:.4f}")

if __name__=='__main__':
    parser=argparse.ArgumentParser('training and evaluation script',parents=[get_args_parser()])
    args=parser.parse_args();main(args)
