import torch
from torch.utils.data import Dataset
import os
import numpy as np
from .common import dataIO, transformData
import glob

io=dataIO() 
transform = transformData()

class Train_Data(Dataset):
    def __init__(self, root_dir, modality_list = ["CT"], patch_size=128):




        self.LQ_paths = []
        self.HQ_paths = []

        for modality in modality_list:

            tmp_paths = glob.glob(os.path.join(root_dir, modality, "train", "LQ", "*.bin"))

            for p in tmp_paths:
                self.LQ_paths.append(p)
                self.HQ_paths.append(p.replace("LQ", "HQ"))

        self.length = len(self.LQ_paths)
        self.label_dict = {
            "PET": 0,
            "LDCT": 1,
            "MRI": 2
            }
        self.patch_size = patch_size


    def __len__(self):
        return self.length

    def analyze_path(self, path):
        path_parts = path.split('/')
        #print(path_parts)
        file_name = path_parts[-1]
        #print(file_name)
        base_name, _ = os.path.splitext(file_name)
        #print(base_name)
        modality = path_parts[-4]
        #print(modality)
        return modality, base_name

    def _ensure_4d_tensor(self, img1, img2):
        """
        确保两个图像张量能够组合成4D张量 [B, C, H, W]
        """
        # 转换为torch张量
        if not isinstance(img1, torch.Tensor):
            img1 = torch.from_numpy(img1)
        if not isinstance(img2, torch.Tensor):
            img2 = torch.from_numpy(img2)

        # 处理不同维度的情况
        def process_single_image(img):
            if len(img.shape) == 2:  # [H, W]
                return img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif len(img.shape) == 3:  # [D, H, W] 或 [C, H, W]
                if img.shape[0] == 1:  # [1, H, W] - 已经有channel维度
                    return img.unsqueeze(0)  # [1, 1, H, W]
                else:  # [D, H, W] - 3D数据，取中间切片
                    middle_slice = img[img.shape[0] // 2]  # [H, W]
                    return middle_slice.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif len(img.shape) == 4:  # [B, C, H, W] - 已经是4D
                return img
            else:
                raise ValueError(f"不支持的图像维度: {img.shape}")

        img1_4d = process_single_image(img1)
        img2_4d = process_single_image(img2)

        # 拼接
        cat_tensor = torch.cat([img1_4d, img2_4d], dim=0)  # [2, 1, H, W]

        return cat_tensor

    def __getitem__(self, idx):

        imgLQ = io.load(self.LQ_paths[idx])
        imgHQ = io.load(self.HQ_paths[idx])

        modality, _ = self.analyze_path(self.LQ_paths[idx])

        imgLQ = transform.normalize(imgLQ, modality)
        imgHQ = transform.normalize(imgHQ, modality)

        # 通用的4D张量处理
        cat_pic = self._ensure_4d_tensor(imgLQ, imgHQ)

        cat_pic = transform.random_crop(tensor=cat_pic, patch_size=[self.patch_size, self.patch_size]).squeeze(1)
        imgLQ, imgHQ = torch.chunk(cat_pic, 2, dim=0)

        class_label = self.label_dict[modality]

        return imgLQ, imgHQ, class_label



class Test_Data(Dataset):
    def __init__(self, root_dir, modality_list = [ "LDCT", "MRI"], use_num = None, target_folder="validation"):
        
        self.LQ_paths = [] 
        self.HQ_paths = []
        #self.save_dir = save_dir
       # os.makedirs(save_dir, exist_ok=True)
        
        
        for modality in modality_list: 
            tmp_paths = glob.glob(os.path.join(root_dir, modality, target_folder, "LQ", "*.nii")) 
            
            use_num = len(tmp_paths) if use_num is None else use_num
            
            for num in range(use_num): 
                p = tmp_paths[num]
                self.LQ_paths.append(p)
                self.HQ_paths.append(p.replace("LQ", "HQ"))  

        self.length = len(self.LQ_paths) 

    def analyze_path(self, path): 
        path_parts = path.split('/') 
        
        file_name = path_parts[-1] 
        base_name, _ = os.path.splitext(file_name) 
        
        modality = path_parts[-4] 
        return modality, base_name
        

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):

       
        imgLQ = io.load(self.LQ_paths[idx])
        imgHQ =io.load(self.HQ_paths[idx]) 
        
        modality, file_name = self.analyze_path(self.LQ_paths[idx])

        imgLQ = transform.normalize(imgLQ, modality) 
        imgHQ = transform.normalize(imgHQ, modality)


        imgLQ = torch.from_numpy(imgLQ).unsqueeze(0) 
        imgHQ = torch.from_numpy(imgHQ).unsqueeze(0)

        return imgLQ, imgHQ, modality, file_name


class DataSampler:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # 如果 DataLoader 中的数据采样完了，重新 shuffle 数据

            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        return batch
