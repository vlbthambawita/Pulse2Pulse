import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import glob


#====== Only for testing ========================
# data_root = "/home/vajira/DL/ecg_test/ECG_data/medians"
# gt_csv = "/home/vajira/DL/ecg_test/ECG_data/ground_truth.csv"
# columns_to_return_gt = ["QOnset", "QOffset", "POnset", "POffset", "TOffset"] 

data_csv = [#"/home/vajira/ecg/GE/asc/ground_truth_updated_vajira.csv", 
 "/home/vajira/ecg/gesus/asc/ground_truth_retired190424_updated.csv",  "/home/vajira/ecg/int99/asc/ground_truth_retired190424_updated.csv"]
data_root = [#"/home/vajira/ecg/GE/asc/medians",  
"/home/vajira/ecg/gesus/asc/medians", "/home/vajira/ecg/int99/asc/medians"]
columns_to_return = ["P_RInterval","QRSDuration",  "Q_TInterval",	"QOnset", "QOffset", "POnset", "POffset", "TOffset"]

class ECGDataALL(Dataset):
    def __init__(self, csv_files, root_dirs, columns_to_return_gt,  transform=None):
        self.dfs = []
        
        for i in range(len(csv_files)):
            df = pd.read_csv(csv_files[i], sep=";") 
            df["root_dir"] = root_dirs[i]
            self.dfs.append(df)
        
        self.df_gt = pd.concat(self.dfs, ignore_index=True, sort=False)  # concatenate all data frames
        self.transform = transform
        self.columns_to_return = columns_to_return_gt

        self.df_gt = self.df_gt.loc[(self.df_gt["ECGCategory"] == "Normal ECG")] # | (self.df_gt["ECGCategory"] == "Otherwise normal ECG")]

        self.df_gt = self.df_gt.dropna(subset=columns_to_return_gt) # droped NaN values
        #self.noise = 0.00000001 # noise to remove floating point exception
        #print(self.df_gt.head())
        #print(self.df_gt.tail())

    def __len__(self):
        return len(self.df_gt)

    def __getitem__(self, idx):
        
        # Get file name of asc file
        asc_file_name = os.path.join(self.df_gt.iloc[idx]["root_dir"],
                                str(self.df_gt.iloc[idx]["TestID"]) + ".asc")
        # print("asc file name=", asc_file_name)
        
        ecg_signals = pd.read_csv(asc_file_name,header=None, sep=" ") # read into dataframe
        ecg_signals = torch.tensor(ecg_signals.values) # convert dataframe values to tensor
        
        # normalization
        ecg_signals = ecg_signals.float()
        #print(ecg_signals)
        
        #print(ecg_signals.shape)
        ecg_signals = ecg_signals  / 6000

        #cropping
        ecg_signals = ecg_signals #[0:4096, :]
        
        # Transposing the ecg signals
        ecg_signals = ecg_signals.t() 
        #ecg_signals = ecg_signals.unsqueeze(0)

       
        
        # reshaping to 1 x 4800 tensor
        #ecg_signals = torch.reshape(ecg_signals, (-1,))
        
        #print("float ecg data")
        #print(ecg_signals)
        
        
        gt = self.df_gt.iloc[idx][self.columns_to_return].values
        #print("gt==", gt)
        
        gt_code = np.zeros(5000) # GT code 

        for gt_value in gt:
            # print("gt value:", gt_value)
            gt_code[int(gt_value)] = 1 # gt_code set to 1 for specific locations

        # normalized gt
        gt = np.array(gt, dtype=np.float)#gt.float()
        gt = gt  # / 600 # length of the time line is 600
       # print(gt)
        
        ########################################################
        # Generating fake ground truth for conditional GAN
        # Based on the rescaled ground truth
        # order of GT => "QOnset", "QOffset", "POnset", "POffset", "TOffset"
        #######################################################
        # Calculated mean and std from all data
        # q_on_set: 	 std = 0.007515963146052018 	 mean = 0.36299921304668137 
        # q_off_set: 	 std = 0.009924477906686631 	 mean = 0.44070541155984194 
        # p_on_set: 	 std = 0.02148843929437315 	     mean = 0.2302298071127185 
        # p_off_set: 	 std = 0.020066791129587362 	 mean = 0.31901371308016874 
        # t_off_set: 	 std = 0.02405158031032503 	     mean = 0.7015729857343781 
        
         # ! these are completely wron  calculations
        # q_on_set: 	 std = 23.616443916502067 	 mean = 159.3095238095238 
        # q_off_set: 	 std = 12.759898314265914 	 mean = 93.24743821579264 
        # p_on_set: 	 std = 28.59356673186787 	 mean = 406.28852722523607 
        # p_off_set: 	 std = 4.509577887631211 	 mean = 217.79952782800885 
        # t_off_set: 	 std = 5.95468674401198 	 mean = 264.4232469359052 

        # Todo Update using these values
        # q_on_set: 	 std = 4.509577887631211 	 mean = 217.79952782800885 
        # q_off_set: 	 std = 5.95468674401198 	 mean = 264.4232469359052 
        # p_on_set: 	 std = 12.893063576623891 	 mean = 138.1378842676311 
        # p_off_set: 	 std = 12.040074677752418 	 mean = 191.40822784810126 
        # t_off_set: 	 std = 14.430948186195016 	 mean = 420.9437914406269 
        ##########################################################
        ##########################################################
        
        # Previous used cases: May be wrong
        # gen_gt_q_on_set = np.random.normal(0.3630, 0.0075, 1)
        # gen_gt_q_off_set = np.random.normal(0.4407, 0.0099, 1)
        # gen_gt_p_on_set = np.random.normal(0.2302, 0.0215, 1)
        # gen_gt_p_off_set = np.random.normal(0.3190, 0.0201, 1)
        # gen_gt_t_off_set = np.random.normal(0.7016, 0.0241, 1)
        
        gen_gt_q_on_set = int(np.random.normal(217.79952782800885, 4.509577887631211,  1))
        gen_gt_q_off_set = int(np.random.normal(264.4232469359052, 5.95468674401198 ,  1))
        gen_gt_p_on_set = int(np.random.normal(138.1378842676311, 12.893063576623891,  1))
        gen_gt_p_off_set = int(np.random.normal(191.40822784810126 , 12.040074677752418,  1))
        gen_gt_t_off_set = int(np.random.normal(420.9437914406269, 14.430948186195016,  1))
        
        # all in one array
        gen_gt = [gen_gt_q_on_set, gen_gt_q_off_set, gen_gt_p_on_set, 
                  gen_gt_p_off_set, gen_gt_t_off_set]
        gen_gt = np.asarray(gen_gt)

        fake_gt_code = np.zeros(5000)

        for gen_gt_value in gen_gt:
            fake_gt_code[gen_gt_value] = 1 
       
       # print(type(gt))
       # print(type(gen_gt))
        
        # Return sample at the end
        sample = {'ecg_signals': ecg_signals}

        if self.transform:
            pass
            #sample = self.transform(sample["ecg_signals"])

        return sample


class ECGDataSimple(Dataset):
    def __init__(self, data_dirs, norm_num=6000, cropping=None, transform=None):
        
        self.all_ecg_files = []
        self.norm_num = norm_num
        self.cropping = cropping
        
        for data_dir in data_dirs:
            ecg_files = glob.glob(data_dir + "/*")
            self.all_ecg_files = self.all_ecg_files + ecg_files

        self.transform = transform

    def __len__(self):
        return len(self.all_ecg_files)

    def __getitem__(self, idx):
        
        # Get file name of asc file
        asc_file_name = self.all_ecg_files[idx] 
        
        ecg_signals = pd.read_csv(asc_file_name,header=None, sep=" ") # read into dataframe
        ecg_signals = torch.tensor(ecg_signals.values) # convert dataframe values to tensor
        
      
        ecg_signals = ecg_signals.float()
        
        #print(ecg_signals.shape)
        ecg_signals = ecg_signals  / self.norm_num # Normalizing aplitude of voltage levels 

        #cropping
        if self.cropping:
            ecg_signals = ecg_signals[self.cropping[0]:self.cropping[1], :]
        
        # Transposing the ecg signals
        ecg_signals = ecg_signals.t() 
        #ecg_signals = ecg_signals.unsqueeze(0)
     
  

        if self.transform:
            ecg_signals = self.transform(ecg_signals)
            #sample = self.transform(sample["ecg_signals"])

        # Return sample at the end
        sample = {'ecg_signals': ecg_signals}

        return sample
    
    
if __name__ == "__main__":
    
    data_roots = [#"/home/vajira/ecg/GE/asc/medians", 
              "/home/vajira/ecg/gesus/asc/rhythm", 
             "/home/vajira/ecg/int99/asc/rhythm"
              ]
    gt_csvs = [#"/home/vajira/ecg/GE/asc/ground_truth.csv", 
           "/home/vajira/ecg/gesus/asc/ground_truth.csv",
          "/home/vajira/ecg/int99/asc/ground_truth.csv"
           ]

    gt_csvs_new = [#"/home/vajira/ecg/GE/asc/ground_truth.csv", 
           "/home/vajira/DL/v243_ground_truth/gesus/ground_truth.csv",
          "/home/vajira/DL/v243_ground_truth/int99/ground_truth.csv"
           ]

    columns_to_return_gt = ["QOnset", "QOffset", "POnset", "POffset", "TOffset"] 
    
    dataset = ECGDataALL(gt_csvs_new, data_roots, columns_to_return_gt)
    print("dataset_size:", len(dataset))
    print("df_gt", dataset.df_gt.head())
# save to csv
    dataset.df_gt.to_csv("gan_trained_data.csv", index=False)

    t =dataset[1]
    print(t)
    print("ecg shape:", t["ecg_signals"].shape)
    print("gt shape:", t["gt"].shape)
    print("GT:",t["gt"])
    print("fake GT:",t["gen_gt"])
    gt = t["gt"].astype(int)
    print(gt)
    #embed = nn.Embedding(600, 100)
    #gt_embed = embed(torch.from_numpy(gt))
    #print(gt_embed)
   # c = gt_embed.reshape(-1)
   # print(c.shape)