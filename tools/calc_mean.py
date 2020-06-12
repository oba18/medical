import sys
sys.path.append('..')
from dataloader import * 
import statistics
from tqdm import tqdm

data = make_data_loader()[0]

def calc_mean(img):
    return img.mean()

def calc_std(img):
    return img.std()

mean_list = []
std_list = []
for batch in tqdm(data):
    img = batch['input'].numpy()
    mean_list += [calc_mean(img)]
    std_list += [calc_std(img)]

m = statistics.mean(mean_list)
s = statistics.mean(std_list)

print (m)
print (s)
