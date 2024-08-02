import timm as tm
import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm 
import copy
import random
vit = tm.create_model('vit_base_patch16_224', pretrained=True)

##control##
print_model = False
target_density = 0.25
mux_size = 6
torch.set_printoptions(precision=2)
torch.set_printoptions(threshold=10_000)
row_size = 32
col_size = 1024
max_conflict =row_size * 0

if print_model:
    for child in vit.children():
        print('=====')
        print(child)

    for name, param in vit.named_parameters():
        print(name, param.shape)

#pruned wgt generation
wgt = vit.get_parameter('blocks.0.attn.qkv.weight')
wgt = wgt.data
a = 9#random.randrange(0,160)
b = 128#random.randrange(0,160)
wgt = wgt[a:a+row_size,b:b+col_size]

mean = torch.mean(wgt)
min_ = torch.min(wgt)
threshold = (mean+min_)/2

density = 1
while (density > target_density):
    threshold = threshold + 0.1e-05   
    pruned_wgt = torch.where(wgt < threshold ,torch.tensor(0) ,1)
    density = torch.count_nonzero(pruned_wgt) / wgt.shape[0] / wgt.shape[1]

print("> Param density:", density," with ",threshold)

#print("\n",pruned_wgt)

#column_group = [[]]

#non-zero count
nz_count = torch.zeros(pruned_wgt.shape[0])
nz_check = torch.where(pruned_wgt > 0, 1, 0)
for r,row in enumerate(nz_check):
    count = torch.bincount(row)
    if count.shape[0] > 1:
        nz_count[r] = count[1]

cols_rows = [[] for i in range(0,pruned_wgt.shape[1])]
for r, row in enumerate(nz_check):
    for c, col in enumerate(row):
        if col > 0:
            cols_rows[c].append(r)

#print("\n",cols_rows)

#column combine
col_group_index = [[0]]
col_group = [cols_rows[0]]
col_group_conflict = [0]
for c in range(1,pruned_wgt.shape[1]):
    if not cols_rows[c]:
        continue
    col_rows = set(cols_rows[c])
    max_density_improve = 0#len(col_rows)
    chosen_group = [-1,[]]
    chosen_conflict = 0
    for g, grp in enumerate(col_group):
        grp_rows = set(grp)
        before_density = len(grp_rows)
        after_density = len(col_rows | grp_rows)
        density_improve = after_density - before_density
        conflict = len(col_rows & grp_rows)
        #conflict = len(col_rows & grp_rows)

        if (conflict+col_group_conflict[g]) <= max_conflict:
            if len(col_group_index[g]) < mux_size:
                if max_density_improve < density_improve:
                    max_density_improve = density_improve
                    chosen_group = [g,list(col_rows | grp_rows)]
                    chosen_conflict = conflict
    
    if chosen_group[0] < 0:
        col_group_index.append([c])
        col_group.append(col_rows)
        col_group_conflict.append(chosen_conflict)
    else:
        col_group_index[chosen_group[0]].append(c)
        col_group[chosen_group[0]] = chosen_group[1]
        col_group_conflict[chosen_group[0]] += chosen_conflict

#print("\n",list(zip(col_group_index,col_group,col_group_conflict)))

packed_matrix_size = pruned_wgt.shape[0] * len(col_group_index)
group_density = sum([len(group) for group in col_group])
group_density = group_density/packed_matrix_size*100

full_nonzeros = sum([len(col) for col in cols_rows])
pruned_nonzeros = sum(col_group_conflict)/full_nonzeros*100

#results
before = torch.full((wgt.shape),-1)
for r,row in enumerate(nz_check):
    for c,check in enumerate(row):
        if check > 0:
            before[r,c]=c

for c in range(0,wgt.shape[1]):
    #print(pruned_wgt[:,c].tolist())
    print(before[:,c].tolist())

after = torch.full((wgt.shape[0],len(col_group)),-1)
for g,cols in enumerate(col_group_index):
    for c,col in enumerate(cols):
        for r in cols_rows[col]:
            after[r,g] = col


for c in range(0,len(col_group)):
    print(c,after[:,c].tolist())

print("\n<<<RESULT>>>\n")
print("> Run Test at Weight ["+str(row_size)+"X",str(col_size),"] with mux:", mux_size,"and conflict: "+str(max_conflict/row_size*100)+"%")
print("\n> Before Column Combining, Density:"+str(density.item()*100)+"%")
print("> After Column Combining, Density: "+str(group_density)+"%")
print("> With Pruned Non_zero Percentage: "+str(round(pruned_nonzeros,3))+"% from "+str(sum(col_group_conflict))+"/"+str(full_nonzeros))

count = 0
for row in after:
    for value in row:
        if value >= 0:
            count+=1

print(after.shape)
print(count,row_size,len(col_group))
print(count/row_size/len(col_group)*100)
