from data import DFaustDataset
from torch.utils.data import DataLoader

 
if __name__ == "__main__":
    data_path = 'outdir/DFaust/DFaust_67'
    
 
    train_dataset = DFaustDataset(data_path, train_flag=True, gt_flag=False, aug_flag=True)
    test_dataset = DFaustDataset(data_path, train_flag=False, gt_flag=False, aug_flag=True)
     
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)

    for item in train_loader:  
        print(item['trans'])