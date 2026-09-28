import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model.DSCENet import DSCENet
from loss.losses import CharbonnierLoss
from evaluation.evaluation_metric import compute_measure
from data.common import transformData, dataIO
from data.MedicalDataUniform import Train_Data, Test_Data, DataSampler
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm
import random
from tools import set_seeds, mkdir
import pdb
import pandas as pd
import matplotlib.pyplot as plt

transformData = transformData()
io = dataIO()
set_seeds(42)

if __name__ == '__main__':
    class ModalGradNormBalancer:
        def __init__(self,
                     num_modalities=3,
                     alpha=1.0,
                     grad_update_freq=10,
                     w_min=0.5,
                     w_max=2.0,
                     eps=1e-8):
            """
            :param num_modalities: 模态数量（默认 3: 0/1/2）
            :param alpha: GradNorm 中控制“偏向落后任务”的指数
            :param grad_update_freq: 每多少个 iteration 更新一次权重（减少开销）
            :param w_min: 权重下限（避免某一模态被压到 0）
            :param w_max: 权重上限（避免某一模态权重爆炸）
            """
            self.num_modalities = num_modalities
            self.alpha = alpha
            self.grad_update_freq = grad_update_freq
            self.w_min = w_min
            self.w_max = w_max
            self.eps = eps

            # 当前权重 w_m
            self.weights = {m: 1.0 for m in range(num_modalities)}
            # 每个模态的初始 loss L_m(0)，在第一次出现该模态时记录
            self.initial_losses = {}  # mod_id -> float
            self.step_count = 0

        def compute_modal_losses(self, restored_main, label_pic, class_label, loss_fn):
            """
            计算当前 batch 内每个模态的重建 loss L_m(t)
            """
            modal_losses = {}
            for mod_id in range(self.num_modalities):
                mod_mask = (class_label == mod_id)
                if mod_mask.sum() > 0:
                    modal_losses[mod_id] = loss_fn(
                        restored_main[mod_mask],
                        label_pic[mod_mask]
                    )
            return modal_losses

        def _maybe_init_losses(self, modal_losses):
            """
            记录每个模态的初始 loss L_m(0)（在该模态第一次出现时）
            """
            for mod_id, loss in modal_losses.items():
                if mod_id not in self.initial_losses:
                    self.initial_losses[mod_id] = float(loss.detach().item())

        def _compute_grad_norms(self, model, optimizer, modal_losses):
            """
            对每个有样本的模态，单独 backward 一次，计算梯度范数 G_m(t)
            """
            grad_norms = {}
            for mod_id, loss_m in modal_losses.items():
                optimizer.zero_grad()
                # 使用当前权重 w_m * L_m 计算梯度，更贴近真实训练中的贡献
                weighted_loss = self.weights.get(mod_id, 1.0) * loss_m
                weighted_loss.backward(retain_graph=True)

                total_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_sq += param_norm.item() ** 2
                grad_norms[mod_id] = total_norm_sq ** 0.5

            optimizer.zero_grad()
            return grad_norms

        def _compute_relative_rates(self, modal_losses):
            """
            计算各模态的相对训练进度 r_m(t)（基于 L_m(t) / L_m(0)）
            """
            ratios = {}
            for mod_id, loss_m in modal_losses.items():
                L0 = self.initial_losses.get(mod_id, float(loss_m.detach().item()))
                current = float(loss_m.detach().item())
                # 相对 loss 比例，带指数 alpha
                ratios[mod_id] = (current / (L0 + self.eps)) ** self.alpha

            if not ratios:
                return {}

            mean_ratio = sum(ratios.values()) / len(ratios)
            rel_rates = {m: (ratios[m] / (mean_ratio + self.eps)) for m in ratios.keys()}
            return rel_rates

        def maybe_update_weights(self, model, optimizer, modal_losses):
            self.step_count += 1
            if not modal_losses:
                return
            # 还没初始化初始 loss 时先记录，不更新权重
            self._maybe_init_losses(modal_losses)
            if len(self.initial_losses) == 0:
                return

            # 不是更新 step，就跳过
            if self.step_count % self.grad_update_freq != 0:
                return

            # 计算每个模态的梯度范数 G_m(t)
            grad_norms = self._compute_grad_norms(model, optimizer, modal_losses)
            if not grad_norms:
                return

            # 计算相对进度 r_m(t)
            rel_rates = self._compute_relative_rates(modal_losses)
            if not rel_rates:
                return

            # 只对当前 batch 内出现的模态进行更新
            active_mods = list(modal_losses.keys())
            # 平均梯度
            G_avg = sum(grad_norms[m] for m in active_mods) / len(active_mods)

            new_weights = self.weights.copy()
            for mod_id in active_mods:
                G_m = grad_norms.get(mod_id, 0.0)
                if G_m <= 0.0:
                    continue
                target_G = rel_rates.get(mod_id, 1.0) * G_avg
                # 让 w_m 往 target_G / G_m 这个比例方向调整
                scale = target_G / (G_m + self.eps)
                new_w = self.weights.get(mod_id, 1.0) * scale
                new_weights[mod_id] = new_w

            # 归一化：让 active 模态的平均权重为 1
            mean_w = sum(new_weights[m] for m in active_mods) / len(active_mods)
            for mod_id in active_mods:
                w = new_weights[mod_id] / (mean_w + self.eps)
                # clip 防止过度失衡
                w = max(self.w_min, min(self.w_max, w))
                self.weights[mod_id] = w

        def get_weighted_loss(self, modal_losses):
            """
            根据当前权重 w_m 组合加权重建 loss
            """
            if not modal_losses:
                return None
            total = 0.0
            for mod_id, loss_m in modal_losses.items():
                w = self.weights.get(mod_id, 1.0)
                total = total + w * loss_m
            return total

    def save_model(G_net_model, save_dir, optimizer_G=None, ex=""):
        save_path = os.path.join(save_dir, "Model")
        mkdir(save_path)
        G_save_path = os.path.join(save_path, 'Generator{}.pth'.format(ex))
        torch.save(G_net_model.cpu().state_dict(), G_save_path)
        G_net_model.cuda()

        if optimizer_G is not None:
            opt_G_save_path = os.path.join(save_path, 'Optimizer_G{}.pth'.format(ex))
            torch.save(optimizer_G.state_dict(), opt_G_save_path)


    def build_train_sampler(modality_list, data_root, batch_size, shuffle=True):
        dataset = Train_Data(root_dir=data_root, modality_list=modality_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=4)
        sampler = DataSampler(dataloader)
        print("data length: \n", dataset.length)
        return sampler


    total_iteration =200000
    val_iteration = 1000

    batch_size =8
    eps = 1e-8
    lr = 2e-4
    psnr_max = 0

    data_root = "/mnt/huxulin/data/all-in-one"
    modality_list = ["LDCT", "MRI","PET"]
    save_dir = "/mnt/huxulin/code_L20/DSCE-Net/result"

    Generator = DSCENet()
    Generator.cuda()

    train_sampler = build_train_sampler(modality_list, data_root, batch_size, shuffle=True)
    valid_loader = DataLoader(Test_Data(root_dir=data_root, use_num=32, modality_list=modality_list), batch_size=1,
                              shuffle=False)

    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)

    lr_scheduler_G = CosineAnnealingLR(optimizer_G, total_iteration, eta_min=1.0e-6)
    L1 = nn.L1Loss()
    cls_criterion = nn.CrossEntropyLoss()
    lambda_cls = 0.01

    modal_balancer = ModalGradNormBalancer(
        num_modalities=3,
        alpha=0.6,
        grad_update_freq=20,
        w_min=0.5,
        w_max=1.5,
        eps=1e-8
    )
    running_loss = []
    eval_metrics = {
        "psnr": [],
        "ssim": [],
        "rmse": []
    }

    pbar = tqdm(total=int(total_iteration))

    print("################ Train ################")
    for iteration in list(range(1, int(total_iteration) + 1)):
     
        l_G = []

        in_pic, label_pic, class_label = next(train_sampler)

        in_pic = in_pic.type(torch.FloatTensor).cuda()
        label_pic = label_pic.type(torch.FloatTensor).cuda()
        class_label = class_label.type(torch.LongTensor).cuda()
        Generator.train()
        optimizer_G.zero_grad()
        restored_main, prompt_logits = Generator(in_pic)
        modal_losses = modal_balancer.compute_modal_losses(restored_main, label_pic, class_label, L1)

        loss_l1 = L1(restored_main, label_pic)

        modal_balancer.maybe_update_weights(Generator, optimizer_G, modal_losses)

        loss_recon = modal_balancer.get_weighted_loss(modal_losses)
        if loss_recon is None:
            loss_recon = loss_l1

        loss_cls = cls_criterion(prompt_logits, class_label)

        loss_G = loss_recon + lambda_cls * loss_cls

        loss_G.backward()
        optimizer_G.step()

        l_G.append(loss_G.item())
        lr_scheduler_G.step()

        if iteration % val_iteration == 0:
            psnr = 0
            ssim = 0
            rmse = 0
            val_loss_sum = 0.0
            Generator.eval()
            for counter, data in enumerate(tqdm(valid_loader)):
                v_in_pic, v_label_pic, modality, file_name = data
                modality = modality[0]
                file_name = file_name[0]

                v_in_pic = v_in_pic.type(torch.FloatTensor).cuda()
                v_label_pic = v_label_pic.type(torch.FloatTensor)
                v_label_norm = v_label_pic.cuda()

                with torch.no_grad():
                    gen_img = Generator(v_in_pic)

                    loss_val = L1(gen_img, v_label_norm)
                val_loss_sum += float(loss_val.item())
                gen_img = transformData.denormalize(gen_img, modality).detach().cpu()

                v_label_pic = transformData.denormalize(v_label_pic, modality)

                gen_img = transformData.truncate_test(gen_img, modality)
                v_label_pic = transformData.truncate_test(v_label_pic, modality)

                data_range = v_label_pic.max() - v_label_pic.min()
                oneEval = compute_measure(gen_img, v_label_pic, data_range=data_range)

                psnr += oneEval[0]
                ssim += oneEval[1]
                rmse += oneEval[2]

                io.save(gen_img.clone().numpy().squeeze(),
                        os.path.join(save_dir, "Gimg", "{}_{}.nii".format(file_name, modality)))


            c_psnr = psnr / (counter + 1)
            c_ssim = ssim / (counter + 1)
            c_rmse = rmse / (counter + 1)
            c_val_loss = val_loss_sum / (counter + 1)
            eval_metrics['psnr'].append(c_psnr)
            eval_metrics['ssim'].append(c_ssim)
            eval_metrics['rmse'].append(c_rmse)

            if c_psnr >= psnr_max:
                psnr_max = c_psnr
                io.save("Best Iteration: {}, PSNR: {}, SSIM:{}, RMSE:{}".format(iteration, c_psnr, c_ssim, c_rmse),
                        os.path.join(save_dir, "best.txt"))
                save_model(G_net_model=Generator, save_dir=save_dir, optimizer_G=optimizer_G, ex="_best")

            csv_path = os.path.join(save_dir, "metrics_log.csv")
            if not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    f.write("iteration,psnr,ssim,rmse\n")
            with open(csv_path, "a") as f:
                f.write(f"{iteration},{c_psnr:.4f},{c_ssim:.4f},{c_rmse:.4f}\n")

        w0 = modal_balancer.weights.get(0, 1.0)
        w1 = modal_balancer.weights.get(1, 1.0)
        w2 = modal_balancer.weights.get(2, 1.0)
        pbar.set_description("loss_l1:{:.6f} w_mod(0:{:.3f}/1:{:.3f}/2:{:.3f}), psnr:{:.6f}".format(
            loss_l1.item(), w0, w1, w2,
            eval_metrics['psnr'][-1] if len(eval_metrics['psnr']) > 0 else 0
        ))
        pbar.update()
