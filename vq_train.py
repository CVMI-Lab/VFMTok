# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import numpy as np
import argparse, pdb
from glob import glob
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from omegaconf import OmegaConf
import torch, os, time, warnings
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from vfmtok.engine.logger import create_logger
from vfmtok.losses.vq_loss import VQLoss
from vfmtok.tokenizer.vq_model import VQ_models
from vfmtok.engine.ema import update_ema, requires_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from vfmtok.engine.distributed import init_distributed_mode
from vfmtok.engine.lr_scheduler import build_scheduler
from vfmtok.data.imagenet_lmdb import ImageNetLmdbDataset as ImageNetDataset
from vfmtok.engine.misc import (is_main_process, all_reduce_mean, concat_all_gather, get_world_size, get_rank)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = get_rank()
    world_size = get_world_size()
    device = rank % torch.cuda.device_count()

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    np.random.seed(os.getpid())

    #* Setup an experiment folder:
    if is_main_process():
        #* Make results folder (holds all experiment subfolders)
        os.makedirs(args.results_dir, exist_ok=True)  
        checkpoint_dir = osp.join(args.results_dir, 'model_dump')
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger_dir = osp.join(args.results_dir, 'logs')
        os.makedirs(logger_dir, exist_ok=True)
        logger = create_logger(logger_dir)
        logger.info(f"Experiment directory created at {args.results_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    transformer_config = OmegaConf.load(args.transformer_config_file)
    #* Create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_slots_embed_dim=args.codebook_slots_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
        transformer_config=transformer_config,
        z_channels=args.z_channels,
    )
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)

    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,  
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    #* Initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    
    #* Setup optimizer
    optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    #* Setup data:
    dataset = ImageNetDataset(args.data_path, args.image_size,  is_train=True)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    warmup_lr_init, warmup_steps = args.warming_up_init_lr, args.warming_up_epochs * len(loader)
    g_sched = build_scheduler(optimizer, args.epochs, len(loader), args.lr, warmup_steps, warmup_lr_init, None)
        
    #* Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        m1, u1 = vq_model.load_state_dict(checkpoint["model"],strict=False)
        if args.ema:
            m1, u1 = ema.load_state_dict(checkpoint["ema"],strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = round(train_steps / int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights

    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0        
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=False)
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    
    model_dirs = osp.realpath(__file__).split('/')
    this_model_dir = '/'.join(model_dirs[2:-1])
    
    if isinstance(vq_model, DDP):
        vq_model.module.freeze_visual_encoder()
    else:
        vq_model.freeze_visual_encoder()
    
    logger.info(f"Training for {args.epochs} epochs, current project dir is {this_model_dir}.")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(loader, dynamic_ncols=True, disable=not is_main_process()) as train_dl:
            for i, (imgs, labels) in enumerate(train_dl):
                
                imgs = imgs.to(device, non_blocking=True)
                
                #* Generator training
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=ptdtype):  
                    recons_imgs, codebook_loss, q_indices = vq_model(imgs)

                    loss_gen, sim_loss = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps+1, 
                                    last_layer=vq_model.module.decoder.last_layer, 
                                    log_every=args.log_every)

                scaler.scale(loss_gen).backward()
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
                
                g_sched.step_update(train_steps + 1)
                scaler.step(optimizer)
                scaler.update()
                if args.ema:
                    update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

                #* Discriminator training            
                optimizer_disc.zero_grad()
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    loss_disc = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1,
                                        log_every=args.log_every)
                scaler_disc.scale(loss_disc).backward()
                if args.max_grad_norm != 0.0:
                    scaler_disc.unscale_(optimizer_disc)
                    torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()
                cur_lr = optimizer.param_groups[0]['lr']
                # # Log loss values:
                torch.cuda.synchronize()
                avg_loss = all_reduce_mean(loss_gen + loss_disc)
                loss_gen = all_reduce_mean(loss_gen)
                loss_discr = all_reduce_mean(loss_disc)
                loss_codebook = all_reduce_mean(sum(codebook_loss))
                loss_sim = all_reduce_mean(sim_loss)
                slot_indices = torch.unique(concat_all_gather(q_indices))
                usage_slot = slot_indices.size(0) / args.codebook_size

                world_size = get_world_size()
                if (train_steps % args.log_every == 0) & is_main_process():

                    train_dl.set_postfix(
                        ordered_dict={
                            "epoch"          : epoch,
                            "iters"          : train_steps,
                            "total_loss"     : avg_loss.item(),
                            "loss_gen"       : loss_gen.item(),
                            "loss_discr"     : loss_discr.item(),
                            'loss_sim'       : loss_sim.item(),
                            'loss_codebook'  : loss_codebook.item(),
                            'usage_slot'     : usage_slot,
                            'n_slots'        : slot_indices.size(0),
                            'learning_rate'  : cur_lr,
                            'this_model_dir' : this_model_dir,
                            "world_size"     : world_size,
                        }
                    )

                #* Save checkpoint:
                if (train_steps > 0) & (train_steps % args.ckpt_every == 0) & is_main_process():
                    if isinstance(vq_model, DDP):
                        model_weight = vq_model.module
                    else:
                        model_weight = vq_model

                    #* Model_weight
                    model_weight = {k:v for k,v in model_weight.state_dict().items() if 'backbone' not in k}
                    checkpoint = {
                            "model": model_weight,
                            "optimizer": optimizer.state_dict(),
                            "discriminator": vq_loss.module.discriminator.state_dict(),
                            "optimizer_disc": optimizer_disc.state_dict(),
                            "steps": train_steps,
                            "args": args
                        }
                    if args.ema:
                        ema_model = {k:v for k,v in ema.state_dict().items() if 'backbone' not in k}
                        checkpoint["ema"] = ema_model
                        
                    ckpt_path = osp.join(checkpoint_dir, 'vit_vqgan_step_{}.pt'.format(train_steps))
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint  to {ckpt_path}")
                train_steps += 1
                dist.barrier()

    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--transformer-config-file", type=str, default='configs/vit_transformer.yaml',)
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-slots-embed-dim", type=int, default=12, help="codebook dimension for queries quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--z-channels", type=int, default=256, help="z-channels")
    
    parser.add_argument("--warming-up-epochs", type=int, default=4, help="warming up iterations")
    parser.add_argument("--warming-up-init-lr", type=float, default=1e-7, help="warming up initial learning rate")
    
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")

    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan', 'dinogan'], default='dinogan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating', 'softplus_g_loss'], default='softplus_g_loss', help="generator loss for gan training")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 336, 384, 512], default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    args = parser.parse_args()
    main(args)
