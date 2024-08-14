import timm as tm
import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm 
import copy
import random
vit = tm.create_model('vit_base_patch16_224', pretrained=True)

##control##
print_model = False
target_density = 0.50
mux_size = 2
torch.set_printoptions(precision=2)
torch.set_printoptions(threshold=10_000)
row_size = 8
col_size = 16
max_conflict = 0#row_size * 0.7
limit = 3

#function
def count_nz(tiles):
    counts = [torch.zeros(tiles[0].shape[1]).int() for i in range(0,len(tiles))]
    for ti, tile in enumerate(tiles):
        for ci, col in enumerate(tile):
            for ri, value in enumerate(col):
                if value >= 0:
                    counts[ti][ri] += 1

    return counts

if print_model:
    for child in vit.children():
        print('=====')
        print(child)

    for name, param in vit.named_parameters():
        print(name, param.shape)

#pruned wgt generation
wgt = vit.get_parameter('blocks.0.attn.qkv.weight')
wgt = wgt.data
a = random.randrange(0,160)
b = random.randrange(0,160)
wgt = wgt[a:a+2*row_size,b:b+col_size]

tiles = [wgt[0:0+row_size, :], wgt[row_size:2*row_size, :]]
pruned_tiles = []
combined_tiles = []

for i, tile in enumerate(tiles):
    mean = torch.mean(tile)
    min_ = torch.min(tile)
    threshold = (mean+min_)/2

    density = 1
    while (density > target_density):
        threshold = threshold + 0.1e-05   
        pruned_tile = torch.where(tile < threshold, 0, 1)
        density = torch.count_nonzero(pruned_tile) / tile.shape[0] / tile.shape[1]
    print("> Tile",i," density:", density," with ",threshold)

    counts = torch.zeros(tile.shape)
    search = torch.where(pruned_tile > 0, 1, 0)
    for j, row in enumerate(search):
        count = torch.bincount(row)
        if count.shape[0] > 1:
            counts[j] = count[1]

    nonzero_i = [[] for j in range(0,pruned_tile.shape[1])]
    for ri, row in enumerate(search):
        for ci, col in enumerate(row):
            if col > 0:
                nonzero_i[ci].append(ri)

    pruned_tiles.append(torch.full(tile.shape,-1))
    for ci, col in enumerate(pruned_tile.transpose(0,1)):
        for r in nonzero_i[ci]:
            pruned_tiles[i][r,ci] = ci

for i, tile in enumerate(pruned_tiles):
    print("\n> Tile",i)
    look = tile.transpose(0,1)
    count = count_nz([look])
    for j, col in enumerate(look):
        print(col,j)
    print(count)

def column_combine(matrix, max_conflict, mux_size):
    search = torch.where(matrix >= 0, 1, 0)
    nonzero_i = [[] for j in range(0,matrix.shape[1])]
    for ri, row in enumerate(search):
        for ci, col in enumerate(row):
            if col > 0:
                nonzero_i[ci].append(ri)

    group_index = [[0]]
    group = [nonzero_i[0]]
    group_conflict = [0]
    for ci in range(1,matrix.shape[1]):
        if not nonzero_i[ci]:
            continue
        col_nonzero_i = set(nonzero_i[ci])
        max_density_improve = 0#len(col_rows)
        chosen_group = [-1,[]]
        chosen_conflict = 0
        for gi, grp in enumerate(group):
            grp_nonzero_i = set(grp)
            before_density = len(grp_nonzero_i)
            after_density = len(col_nonzero_i | grp_nonzero_i)
            density_improve = after_density - before_density
            conflict = len(col_nonzero_i & grp_nonzero_i)
            #conflict = len(col_rows & grp_rows)

            if (conflict+group_conflict[gi]) <= max_conflict:
                if len(group_index[gi]) < mux_size:
                    if max_density_improve < density_improve:
                        max_density_improve = density_improve
                        chosen_group = [gi,list(col_nonzero_i | grp_nonzero_i)]
                        chosen_conflict = conflict

        if chosen_group[0] < 0:
            group_index.append([ci])
            group.append(col_nonzero_i)
            group_conflict.append(chosen_conflict)
        else:
            group_index[chosen_group[0]].append(ci)
            group[chosen_group[0]] = chosen_group[1]
            group_conflict[chosen_group[0]] += chosen_conflict

    packed_matrix = torch.full((len(group),matrix.shape[0]),-1)
    for gi, cols in enumerate(group_index):
        for ci in cols:
            for ri in nonzero_i[ci]:
                packed_matrix[gi,ri] = ci

    return packed_matrix, group_index

combined_matrix, recipe = column_combine(pruned_tiles[0],0,4)

look = combined_matrix

def scatter_column(tile):
    table = tile.clone()
    not_used = [i for i in range(0,table.shape[1])]
    result = []
    
    while not_used:
        look = table[:,not_used]
        search = [0 for i in range(0,look.shape[1])]

        for row in look:
            for ci, value in enumerate(row):
                if value >= 0:
                    search[ci] += 1

        d_ci = search.index(max(search))
        dense_col = look[:,d_ci]
        block = []
        for ci, value in enumerate(dense_col):
            if value >= 0:
                block.append(ci)

        count = [0 for i in range(0,look.shape[0])]

        for ri, row in enumerate(look):
            for ci, value in enumerate(row):
                if value < 0 and ci != d_ci:
                    count[ri]+=1

        slot = []

        able = True
        for b in block:
            if count[b] == 0:
                able = False
        if able:
            group = [d_ci]
           
            while not (set(block).issubset(set(slot))):
                look_ = look.transpose(0,1)
                choice = None
                max_nslot = 0
                max_crit = 0

                slot_tmp = []
                length = 0
                conflict = 0
                choice = None
                need = (set(block)-set(slot))
                for ci, col in enumerate(look_):
                    if ci not in group:
                        for ri,value in enumerate(col):
                            if value < 0:
                                slot_tmp.append(ri)
                        if len(set(slot_tmp) & need) >= length:
                            if len(set(slot_tmp) & need) > length:
                                choice = ci
                                length = len(set(slot_tmp) & need)
                                conflict = len(set(slot_tmp) & set(slot))

                            elif len(set(slot_tmp) & need) == length and len(set(slot_tmp) & set(slot)) < conflict:
                                choice = ci
                                length = len(set(slot_tmp) & need)
                                conflict = len(set(slot_tmp) & set(slot))

                group.append(choice)
                for ri, value in enumerate(look_[choice]):
                    if value < 0 and ri not in slot:
                        slot.append(ri)
            result.append([not_used[g] for g in group])

            not_used = list(set(not_used)-set([not_used[g] for g in group]))

        else:
            break

    return result

result = scatter_column(pruned_tiles[0])
for group in result:
    print("\n")
    for col in group:
        print(pruned_tiles[0][:,col])

print(result)
