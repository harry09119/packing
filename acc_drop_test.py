from my_lib import column_combine,pruned_column_scatter, counting

import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch
from torch.nn import functional as f
import copy
from torch.utils.data import DataLoader
from collections import OrderedDict

device = "cuda" if torch.cuda.is_available() else "cpu"

tiling = True
with_dcs = True
loop = 50
mux_size = 4
tile_size = 16
path = '/home/harry09119/packing/final/model'
file_name = '/sparse_resnet50_s80.pt'

resnet = models.resnet50(pretrained = False)
#resnet = torch.load(path+file_name,map_location=device)
#resnet.load_state_dict(path + file_name)
checkpoint = torch.load(path+file_name,map_location=device)
checkpoint_re = OrderedDict() 
for key,tensor in checkpoint.items():
    checkpoint_re[".".join(key.split(".")[1:])] = tensor

#for key,_ in checkpoint_re.items():
#    print(key)

resnet.load_state_dict(checkpoint_re)
resnet.eval()
#pretrained_dict = resnet.state_dict()

checkpoint_modified = copy.deepcopy(checkpoint_re)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

imagenet = torchvision.datasets.ImageNet('/data/imagenet',split='val',transform=transform)
test_dataloader = DataLoader(imagenet, batch_size=16, shuffle=False)

with torch.no_grad():
    correct_top1 = 0
    correct_top5 = 0
    total = 0
   
    for idx,(images,labels) in enumerate(test_dataloader):
        if idx > loop:
            break
        outputs = resnet(images)
    
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct_top1 += (pred == labels).sum().item()
        
        _, rank5 = outputs.topk(5, 1, True, True)
        rank5 = rank5.t()
        correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))
        
        for k in range(6):
            correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

        correct_top5 += correct_k.item()

        print("step : {} / {}".format(idx + 1, len(imagenet)/int(labels.size(0))))
        #print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
                                                                     
        #print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))

    print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
    print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))
    
    top1_before = round(correct_top1 / total * 100,2)
    top5_before = round(correct_top5 / total * 100,2)

    total_nz = 0
    total_sparsity = 0
    layer_num = 0
    total_params = 0

    total_pruned = 0
    for layer in list(resnet.named_parameters()):
        if "conv" in layer[0]:
            layer_num +=1
            print("Modifying Layer",layer_num,"-",layer[0], layer[1].shape)
            #print(layer[0],":",layer[1].detach().shape)
            
            sample = copy.deepcopy(layer[1].detach())
            base_shape = torch.tensor(list(sample.shape))
            
            if not tiling:
                sample = sample.reshape([sample.shape[0],-1])
                look = torch.where(sample == 0,-1, 1)
                edited_matrix, mux_used, pruned = column_combine( look, sample.shape[0]*0.25, 4)
                for where in pruned:
                    sample[where[0],where[1]] = 0
                
                params = torch.prod(base_shape)
                nz_before = ((look > 0).count_nonzero()).item()
                sparsity = (1 - nz_before/params).item()
                nz_after = ((edited_matrix >= 0).count_nonzero()).item()
                sparsity -= 1 - nz_after/(len(mux_used)*sample.shape[0])

                check = (len(pruned) == (nz_before - nz_after))
                nz_pruned = ((nz_before - nz_after)/nz_before)*100
                print("> Sparsity Decrease: ", str(round(nz_before*100,2))+"%","to",str(round(sparsity*100,2))+"%")
                print("> Pruned: ", round(nz_pruned,2), check)

            else:
                sample = sample.reshape([int(sample.shape[0]/tile_size),tile_size,-1])
                look = torch.where(sample == 0,-1, 1)
                
                params = torch.prod(base_shape)
                nz_before = (look > 0).count_nonzero()
                sparsity = (1 - nz_before/params).item()
                if round(sparsity,1) == 0:
                    continue

                pruned_len = 0
                matrix_len = 0
                nz_tiles = 0
                for ti in range(0, sample.shape[0]):
                    edited_matrix, mux_used, pruned = column_combine(look[ti], tile_size*0, mux_size/2)
                    
                    if not with_dcs:
                        nz_tiles += (edited_matrix >= 0).count_nonzero()
                        pruned_len += len(pruned)
                        matrix_len += edited_matrix.shape[1]
                        for where in pruned:
                            sample[ti][where[0],where[1]] = 0

                    else:
                        edited_matrix, pruned = pruned_column_scatter(edited_matrix.transpose(0,1), mux_size, 0.25, mux_used)
                        nz_tiles += (edited_matrix >= 0).count_nonzero()
                        pruned_len += len(pruned)
                        matrix_len += edited_matrix.shape[1]
                        for where in pruned:
                            sample[ti][where[0],where[1]] = 0

                params = (torch.prod(base_shape)).item()
                nz_before = ((look > 0).count_nonzero()).item()
                sparsity = 1 - nz_before/params
                nz_after = nz_tiles.item()
                sparsity -= 1 - nz_after/(matrix_len*tile_size)

                check = (pruned_len == (nz_before - nz_after))
                nz_pruned = ((nz_before - nz_after)/nz_before)*100
                print("> Sparsity Decrease: ", str(round(nz_before*100,2))+"%","to",str(round(sparsity*100,2))+"%")
                print(nz_before, nz_after)
                print("> Pruned: ", round(nz_pruned,2), check)
                if not check:
                    print(">> Error: ",nz_before, len(pruned))
           
            sample = sample.reshape(layer[1].shape)
            checkpoint_modified[layer[0]] = sample
    
    print(((total_nz/total_params*100),2))

    resnet.load_state_dict(checkpoint_modified)
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for idx,(images,labels) in enumerate(test_dataloader):
        if idx > loop:
            break
        outputs = resnet(images)
    
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct_top1 += (pred == labels).sum().item()
        
        _, rank5 = outputs.topk(5, 1, True, True)
        rank5 = rank5.t()
        correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))
        
        for k in range(6):
            correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

        correct_top5 += correct_k.item()

        print("step : {} / {}".format(idx + 1, len(imagenet)/int(labels.size(0))))
        print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
                                                                     
        print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))

    print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
    print("top-5 percentage :  {0:0.2f}%".format(correct_top5 / total * 100))

    top1_after = round(correct_top1 / total * 100,2)
    top5_after = round(correct_top5 / total * 100,2)   
    
    print("\nAccuracy Drop")
    print(top1_before, " to",top1_after)
    print(top5_before, " to",top5_after)
    
    print(total_pruned/total_nz*100)

