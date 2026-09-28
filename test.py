import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model.Restormer_MDTA_gai5 import Restormer
from evaluation.evaluation_metric import compute_measure
from data.common import transformData, dataIO
from data.MedicalDataUniform import Test_Data
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb
import pandas as pd

transformData = transformData()
io = dataIO()

data_root = "/mnt/huxulin/data/all-in-one"
modality_list =  ["LDCT","PET","MRI" ]
save_dir = "/mnt/huxulin/code_L20/DSCE- Net/result"

Generator = Restormer()
Generator.cuda()

Generator.load_state_dict(torch.load(os.path.join(save_dir, "Model", "Generator_best.pth")))
Generator.eval()
overall_psnr = []
overall_ssim = []
overall_rmse = []
for modality_name in modality_list:
    test_loader = DataLoader(Test_Data(root_dir=data_root, modality_list=[modality_name], target_folder="test"),
                             batch_size=1, shuffle=False, num_workers=4)
    psnr_list = []
    ssim_list = []
    rmse_list = []
    name_list = []
    for counter, data in enumerate(tqdm(test_loader)):
        v_in_pic, v_label_pic, modality, file_name = data
        a = file_name
        modality = modality[0]
        file_name = file_name[0]

        v_in_pic = v_in_pic.type(torch.FloatTensor).cuda()
        v_label_pic = v_label_pic.type(torch.FloatTensor)

        with torch.no_grad():
            gen_img = Generator(v_in_pic)
        # gen_img = gen_img[-2]
        gen_img = transformData.denormalize(gen_img, modality).detach().cpu()

        v_label_pic = transformData.denormalize(v_label_pic, modality)
        # print("gen_img shape:", gen_img.shape)
        # print("label shape:", v_label_pic.shape)

        '''
        truncation for test_hair image 
        CT:[-160, 240]
        '''

        gen_img = transformData.truncate_test(gen_img, modality)
        v_label_pic = transformData.truncate_test(v_label_pic, modality)

        data_range = v_label_pic.max() - v_label_pic.min()
        oneEval = compute_measure(gen_img, v_label_pic, data_range=data_range)

        psnr_list.append(oneEval[0])
        ssim_list.append(oneEval[1])
        rmse_list.append(oneEval[2])
        name_list.append(file_name)

        io.save(gen_img.clone().numpy().squeeze(),
                os.path.join(save_dir, "test_result", modality, "{}.nii".format(file_name)))

    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    rmse_list = np.array(rmse_list)
    name_list = np.array(name_list)
    c_psnr = psnr_list.mean()
    c_ssim = ssim_list.mean()
    c_rmse = rmse_list.mean()
    # variance / std across test cases (sample statistics)
    v_psnr = psnr_list.var(ddof=1) if psnr_list.size > 1 else 0.0
    v_ssim = ssim_list.var(ddof=1) if ssim_list.size > 1 else 0.0
    v_rmse = rmse_list.var(ddof=1) if rmse_list.size > 1 else 0.0
    s_psnr = np.sqrt(v_psnr)
    s_ssim = np.sqrt(v_ssim)
    s_rmse = np.sqrt(v_rmse)

    print(
        " ^^^Final Test  {}   psnr:{:.6} (var:{:.6}, std:{:.6}), "
        "ssim:{:.6} (var:{:.6}, std:{:.6}), rmse:{:.6} (var:{:.6}, std:{:.6}) ".format(
            modality_name, c_psnr, v_psnr, s_psnr, c_ssim, v_ssim, s_ssim, c_rmse, v_rmse, s_rmse
        )
    )
    overall_psnr.append(c_psnr)
    overall_ssim.append(c_ssim)
    overall_rmse.append(c_rmse)

    summary_line = (
        "Final Test {}   psnr:{:.6} (var:{:.6}, std:{:.6}), "
        "ssim:{:.6} (var:{:.6}, std:{:.6}), rmse:{:.6} (var:{:.6}, std:{:.6})\n".format(
            modality_name, c_psnr, v_psnr, s_psnr, c_ssim, v_ssim, s_ssim, c_rmse, v_rmse, s_rmse
        )
    )
    with open(os.path.join(save_dir, "test_result", "{}_final_test.txt".format(modality_name)), "a") as f:
        f.write(summary_line)
    with open(os.path.join(save_dir, "test_result", "ALL_final_test.txt"), "a") as f:
        f.write(summary_line)
    result_dict = {
        "NAME": name_list,
        "PSNR": psnr_list,
        "SSIM": ssim_list,
        "RMSE": rmse_list,
    }
    result = pd.DataFrame({key: pd.Series(value) for key, value in result_dict.items()})
    result.to_csv(os.path.join(save_dir, "test_result", "{}_result.csv".format(modality_name)))

# overall average across modalities
if len(overall_psnr) > 0:
    avg_psnr = float(np.mean(overall_psnr))
    avg_ssim = float(np.mean(overall_ssim))
    avg_rmse = float(np.mean(overall_rmse))
    print(" === Overall (LDCT+MRI+PET) ===  psnr:{:.6f}, ssim:{:.6f}, rmse:{:.6f}".format(avg_psnr, avg_ssim, avg_rmse))
    with open(os.path.join(save_dir, "test_result", "ALL_final_test.txt"), "a") as f:
        f.write("\n")
        f.write("Overall (LDCT+MRI+PET)  psnr:{:.6f}, ssim:{:.6f}, rmse:{:.6f}\n".format(avg_psnr, avg_ssim, avg_rmse))
