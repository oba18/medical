from torch.utils.data import DataLoader

from dataloader.dataset import Dataset
from dataloader.get_lidc import Lidc

def make_data_loader(batch_size=16, is_develop=False):
    
    """
    Prepare Dataset and apply Dataloader.
    You don't have to change it.
    """
    lidc = Lidc(is_develop=is_develop)
    il = lidc.img_list
    ml = lidc.mask_list
    train_set = Dataset(il, ml, split="train")
    val_set = Dataset(il, ml, split="val")
    test_set = Dataset(il, ml, split="test")
    num_class = train_set.NUM_CLASSES
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_class
