#===========================================
# Reference: https://github.com/mazzzystar/WaveGAN-pytorch
#===========================================

import argparse
import os
import yaml
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np

#Pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import models, transforms
from torchvision.utils import save_image
import wandb
from torch.autograd import Variable
from torchinfo import summary
from torch import autograd


# Model specific

from data.ecg_data_loader import ECGDataSimple as ecg_data
from data.ecg_data_loader import PTBXLDataset
from models.pulse2pulse import WaveGANGenerator as Pulse2PuseGenerator
from models.pulse2pulse import WaveGANDiscriminator as Pulse2PulseDiscriminator
from utils.utils import calc_gradient_penalty, get_plots_RHTM_10s, get_plots_all_RHTM_10s

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

opt = SimpleNamespace(**cfg)
print(opt)

#==========================================
# Device handling
#==========================================
torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device=", device)

#===========================================
# Folder handling
#===========================================

#make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)


# make subfolder in the output folder 
checkpoint_dir = os.path.join(opt.out_dir, opt.exp_name + "/checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

#==========================================
# Weights & Biases
#==========================================
wandb.init(project=opt.wandb_project, name=opt.exp_name, config=vars(opt))


#==========================================
# Prepare Data
#==========================================
def prepare_data():
    if opt.dataset == "ptbxl":
        train_dataset = PTBXLDataset(opt.ptbxl_path, split='train',
                                     sampling_rate=opt.ptbxl_sampling_rate)
        val_dataset   = PTBXLDataset(opt.ptbxl_path, split='val',
                                     sampling_rate=opt.ptbxl_sampling_rate)
        test_dataset  = PTBXLDataset(opt.ptbxl_path, split='test',
                                     sampling_rate=opt.ptbxl_sampling_rate)

        print("PTBXL - Train size:", len(train_dataset),
              " Val size:", len(val_dataset),
              " Test size:", len(test_dataset))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,
                                                   shuffle=True, num_workers=8)
        val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=opt.bs,
                                                   shuffle=False, num_workers=8)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,
                                                   shuffle=False, num_workers=8)
        return {"train": train_loader, "val": val_loader, "test": test_loader}
    else:
        dataset = ecg_data(opt.data_dirs, norm_num=6000, cropping=None, transform=None)
        print("Dataset size=", len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs,
                                                 shuffle=True, num_workers=8)
        return {"train": dataloader, "val": None, "test": None}

#===============================================
# Prepare models
#===============================================
def prepare_model():
    netG = Pulse2PuseGenerator(model_size=opt.model_size, ngpus=opt.ngpus, upsample=True)
    netD = Pulse2PulseDiscriminator(model_size=opt.model_size, ngpus=opt.ngpus)

    netG = netG.to(device)
    netD = netD.to(device)

    return netG, netD

#====================================
# Run training process
#====================================
def run_train():
    netG, netD = prepare_model()
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    dataloaders = prepare_data()
    train(netG, netD, optimizerG, optimizerD, dataloaders["train"])

def train(netG, netD, optimizerG, optimizerD, dataloader):

    for epoch in tqdm(range(opt.start_epoch + 1, opt.start_epoch + opt.num_epochs + 1)):

        len_dataloader = len(dataloader)
        #print("Length of Dataloader:", len_dataloader)

        train_G_flag = False
        D_cost_train_epoch = []
        D_wass_train_epoch = []
        G_cost_epoch = []
    
        for i, sample in tqdm(enumerate(dataloader, 0)):

            if (i+1) % 5 == 0:
                train_G_flag = True



            # Set Discriminator parameters to require gradients.
            #print(train_G_flag)
            for p in netD.parameters():
                p.requires_grad = True

            #one = torch.Tensor([1]).float()
            one = torch.tensor(1, dtype=torch.float)
            neg_one = one * -1

            one = one.to(device)
            neg_one = neg_one.to(device)

            #############################
            # (1) Train Discriminator
            #############################

            real_ecgs = sample["ecg_signals"].to(device)
            #print("real ecgs shape", real_ecgs.shape)
            b_size = real_ecgs.size(0)

            netD.zero_grad()


            # Noise
            noise = torch.Tensor(b_size, 8, 5000).uniform_(-1, 1)
            noise = noise.to(device)
            noise_Var = Variable(noise, requires_grad=False)

            # real_data_Var = numpy_to_var(next(train_iter)['X'], cuda)

            # a) compute loss contribution from real training data
            D_real = netD(real_ecgs)
            D_real = D_real.mean()  # avg loss
            D_real.backward(neg_one)  # loss * -1

            # b) compute loss contribution from generated data, then backprop.
            fake = autograd.Variable(netG(noise_Var).data)
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty(netD, real_ecgs,
                                                    fake.data, b_size, opt.lmbda,
                                                    use_cuda=True)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator.
            optimizerD.step()

            D_cost_train_cpu = D_cost_train.data.cpu()
            D_wass_train_cpu = D_wass_train.data.cpu()


            D_cost_train_epoch.append(D_cost_train_cpu)
            D_wass_train_epoch.append(D_wass_train_cpu)


            #############################
            # (3) Train Generator
            #############################
            if train_G_flag:
                # Prevent discriminator update.
                for p in netD.parameters():
                    p.requires_grad = False

                # Reset generator gradients
                netG.zero_grad()

                # Noise
                noise = torch.Tensor(b_size, 8, 5000).uniform_(-1, 1)
                
                noise = noise.to(device)
                noise_Var = Variable(noise, requires_grad=False)

                fake = netG(noise_Var)
                G = netD(fake)
                G = G.mean()

                # Update gradients.
                G.backward(neg_one)
                G_cost = -G

                optimizerG.step()

                # Record costs
                #if cuda:
                G_cost_cpu = G_cost.data.cpu()
                #print("g_cost=",G_cost_cpu)
                G_cost_epoch.append(G_cost_cpu)
                #print("Epoch{} - {}_G_cost_cpu:{}".format(epoch, i, G_cost_cpu))
                #G_cost_epoch.append(G_cost_cpu.data.numpy())
                train_G_flag =False

                #print("real ecg:", real_ecgs.shape)
                #print("fake ecg:", fake.shape)
            if i == 0: # take the first batch to plot
                real_ecgs_to_plot = real_ecgs
                fake_to_plot = fake
            #    break
        #print(G_cost_epoch)

        D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
        D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
        G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

        
        wandb.log({
            "D_cost_train": D_cost_train_epoch_avg,
            "D_wass_train": D_wass_train_epoch_avg,
            "G_cost": G_cost_epoch_avg,
        }, step=epoch)

        print("Epochs:{}\t\tD_cost:{}\t\t D_wass:{}\t\tG_cost:{}".format(
                    epoch, D_cost_train_epoch_avg, D_wass_train_epoch_avg, G_cost_epoch_avg))

        # Save model
        if epoch % opt.checkpoint_interval == 0:
            save_model(netG, netD, optimizerG, optimizerD, epoch)
            fig = get_plots_RHTM_10s(real_ecgs_to_plot[0].detach().cpu(), fake_to_plot[0].detach().cpu())
            fig_2 = get_plots_all_RHTM_10s(real_ecgs_to_plot.detach().cpu(), fake_to_plot.detach().cpu())

            wandb.log({"sample": wandb.Image(fig), "sample_batch": wandb.Image(fig_2)}, step=epoch)


#=====================================
# Save models
#=====================================
def save_model(netG, netD, optimizerG, optimizerD,  epoch):
   
    check_point_name = py_file_name + "_epoch:{}.pt".format(epoch) # get code file name and make a name
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model
    torch.save({
        "epoch": epoch,
        "netG_state_dict": netG.state_dict(),
        "netD_state_dict": netD.state_dict(),
        "optimizerG_state_dict": optimizerG.state_dict(),
        "optimizerD_state_dict": optimizerD.state_dict(),
        # "train_loss": train_loss,
        #"val_loss": validation_loss
    }, check_point_path)

#====================================
# Re-train process
#====================================
def run_retrain():
    print("run retrain started........................")
    netG, netD = prepare_model()

    #netG.cpu()
    #netD.cpu()

    # loading checkpoing
    chkpnt = torch.load(opt.checkpoint_path, map_location="cpu")

    netG.load_state_dict(chkpnt["netG_state_dict"])
    netD.load_state_dict(chkpnt["netD_state_dict"])

    netG = netG.to(device)
    netD = netD.to(device)

    print("model loaded from checkpoint=", opt.checkpoint_path)

    # setup start epoch to checkpoint epoch
    opt.__setattr__("start_epoch", chkpnt["epoch"])

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    dataloaders = prepare_data()
    train(netG, netD, optimizerG, optimizerD, dataloaders["train"])


#=====================================
# Check model
#====================================
def check_model_graph():
    netG, netD = prepare_model()
    print(netG)
    netG = netG.to(device)
    netD = netD.to(device)

    summary(netG, (8,5000))
    summary(netD, (8, 5000))




if __name__ == "__main__":

    data_loaders = prepare_data()
    print(vars(opt))
    print("Train size:", len(data_loaders["train"].dataset))
    print("Test OK")

    # Train or retrain or inference
    if opt.action == "train":
        print("Training process is strted..!")
        run_train()
        pass
    elif opt.action == "retrain":
        print("Retrainning process is strted..!")
        run_retrain()
        pass
    elif opt.action == "inference":
        print("Inference process is strted..!")
        pass
    elif opt.action == "check":
        check_model_graph()
        print("Check pass")

    wandb.finish()
