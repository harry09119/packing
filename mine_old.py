import timm as tm
import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm 
vit = tm.create_model('vit_base_patch16_224', pretrained=True)

##control##
print_model = False
target_density = 0.25
mux_size = 4
torch.set_printoptions(precision=2)

if print_model:
    for child in vit.children():
        print('=====')
        print(child)

    for name, param in vit.named_parameters():
        print(name, param.shape)

#pruned wgt generation
wgt = vit.get_parameter('blocks.0.attn.qkv.weight')
wgt = wgt.data
wgt = wgt

mean = torch.mean(wgt)

density = 1
while (density > target_density):
    pruned_wgt = torch.where(wgt < mean ,torch.tensor(0.) ,wgt)
    density = torch.count_nonzero(pruned_wgt) / wgt.shape[0] / wgt.shape[1]
    mean = mean + 1.0e-05

print("> Param density:", density," with ",mean)
pruned_wgt = pruned_wgt[12:20,130:146]

print("\n",pruned_wgt)

#non-zero count
nz_count = torch.zeros(pruned_wgt.shape[0])
nz_check = torch.where(pruned_wgt > 0, 1, 0)

for r,row in enumerate(nz_check):
    count = torch.bincount(row)
    if count.shape[0] > 1:
        nz_count[r] = count[1]

print("\n> nz_count\n",nz_count)

crit_len = int(max(nz_count).item())
crit_ind = torch.argmax(nz_count)
print("\n> ", wgt.shape[1]," cols are packed into ",crit_len," cols in ",crit_len )

nz_cols = torch.full((pruned_wgt.shape[0], crit_len),-1)

for i, row in enumerate(nz_check):
    nz_cols_ = row.nonzero()
    for j,nz_i in enumerate(nz_cols_):
        nz_cols[i][j] = nz_i

print("\n",nz_cols)

nz_index = nz_check.nonzero(as_tuple=True)
crit_cols = nz_cols[crit_ind]
print("\n> critical_path \n",crit_cols)

#Input Plan Generation
union_inp = torch.unique(nz_cols, sorted=True)
union_inp = union_inp[union_inp != -1]
max_inp = max(union_inp)
all_inp = torch.range(0,max_inp+1)
print("\n",union_inp)

inp_plan = torch.zeros(crit_cols.shape[0],mux_size)
for i in range(0, crit_cols.shape[0]):
    base = crit_cols[i]
#    (all_inp == crit_cols[i]).nonzero(as_tuple= True)[0]
    start = base - mux_size + 1
    for j in range(0,mux_size):
        look = start + j
        if look < 0:
            look = max_inp + look
        elif look > max_inp:
            look = look%(all_inp.shape[0])
        inp_plan[i][j] = all_inp[look]

print("\n",inp_plan)

#n:m like packing
"""
packed_wgt = torch.zeros(pruned_wgt.shape[0],crit_len)
old = 0
c = -1
for i, r in enumerate(nz_index[0]):
    new = r
    if old < new:
        c = 0
    else:
        c += 1
    packed_wgt[r.item()][c] = pruned_wgt[r.item()][nz_index[1][i]]
    old = new

print("\n",packed_wgt)
"""
"""
#non_zero allocate
packed_wgt = torch.zeros(pruned_wgt.shape[0],crit_len)
packed_ind = torch.full((pruned_wgt.shape[0],crit_len), -1)

able_cols = []#nz_cols.clone().detach()
for row in nz_cols:
    cols = row.unique(sorted=True)
    able_cols.append(cols[cols != -1])
_, row_order = torch.sort(nz_count, descending=True)

for r in row_order:
    cols = able_cols[r]
    for oc,look in enumerate(inp_plan):
        for c in range(0,len(cols)):
            if cols[c] in look:
                cols = torch.cat((cols[:c],cols[c+1:]),0)
                break
    
    if len(cols.size()) > 0:

                packed_wgt[r][oc] = pruned_wgt[r][cols[c]]
                packed_ind[r][oc] = cols[c]
       

print(able_cols)


sparsity = 0
for e in torch.flatten(packed_ind):
    if e == -1:
        sparsity += 1
sparsity = sparsity/pruned_wgt.shape[0]/crit_len * 100
print(packed_ind)
print("\n",packed_wgt)
print("> Sparsity become 25% to "+str(sparsity)+"%")
"""
