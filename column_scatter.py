import timm as tm
import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm 
import copy
import random
import math
vit = tm.create_model('vit_base_patch16_224', pretrained=True)

##control##
debug = True
print_model = False
target_density = 0.2
mux_size = 4
torch.set_printoptions(precision=2)
torch.set_printoptions(threshold=10_000)
row_size = 8
col_size = 16
max_conflict = 0#row_size * 0.7
limit = 3

#pruned wgt generation
wgt = vit.get_parameter('blocks.0.attn.qkv.weight')
wgt = wgt.data
a = 99#random.randrange(0,160)
b = 99#random.randrange(0,160)
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

def counting(matrix):
    nonzero = 0
    zero = 0
    for col in matrix:
        for value in col:
            if value < 0:
                zero += 1
            else:
                nonzero += 1
    return nonzero, zero

def row_balancing(matrix):
    look = matrix.clone().transpose(0,1)

    modify = torch.full(look.shape,-1)

    row_count = [0 for x in look[0]]
    for ci, col in enumerate(look):
        for ri, value in enumerate(col):
            if value>=0:
                row_count[ri] += 1

    row_order = sorted(range(len(row_count)), key=lambda k: row_count[k])

    total = 0
    for value in row_count:
        total += value

    custom_order = []

    for ri in range(0, math.trunc(len(row_order)/2)):
        custom_order.append(row_order[ri])
        custom_order.append(row_order[len(row_order)-1-ri])

    if len(row_order)%2 == 1:
        custom_order.append(math.trunc(len(row_order)/2+1))
    
    look = matrix[custom_order].transpose(0,1)
    
    col_count = [0 for x in matrix[0]]
    for ci, col in enumerate(look):
        for ri, value in enumerate(col):
            if value>=0:
                col_count[ci]+=1

    col_sort_i = sorted(range(len(col_count)), key=lambda k: col_count[k])

    #balancing
    for ci in col_sort_i:
        row_count = [0 for x in look[0]]
        for col in look:
            for i, value in enumerate(col):
                if value>=0:
                    row_count[i]+=1
        
        avg = round(total/len(row_count))
       
        for ri, value in enumerate(look[ci]):
            if ri > 0 and modify[ci,ri] < 0:
                if value >= 0 and look[ci,ri-1] < 0:
                    if row_count[ri] >= avg and row_count[ri-1] < avg:
                        tmp = look[ci,ri-1]
                        look[ci,ri-1] = value
                        look[ci,ri] = -1
                        row_count[ri] -= 1
                        row_count[ri-1] += 1
                        modify[ci,ri] = 1

    #print(look)
    row_count = [0 for x in look[0]]
    for col in look:
        for i, value in enumerate(col):
            if value>=0:
                row_count[i]+=1
    #print(row_count)
    return look.transpose(0,1), modify.transpose(0,1), row_count

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
    
    group_len = [len(group) for group in group_index]

    return packed_matrix.transpose(0,1), group_len
       
def pruned_column_scatter(matrix, max_cols, max_conflict, group_len):
    col_list = [[ci,col] for ci, col in enumerate(matrix)]
    col_nonz = []
    for ci, col in col_list:
        count = 0
        for value in col:
            if value >= 0:
                count += 1

        col_nonz.append(count)
   
    d_list = [x[0] for n, x in sorted(zip(col_nonz, col_list),reverse=True) if n > 0]
    s_list = [x[0] for n, x in sorted(zip(col_nonz, col_list)) if n > 0]
    
    used = []

    result = []
    pruned = []
    for d_ci in d_list:
        block = []
        count = 0
        for ri, value in enumerate(col_list[d_ci][1]):
            if value >= 0:
                count += 1
                block.append(ri)
        
        if not block:
            continue
        if d_ci in used:
            continue

        slot = []
        
        s_group = []
        for l in range(0,max_cols):
            choice = [None, 0, [], 0]
            need= set(block)-set(slot)
            if not need:
                break
            
            elif len(need) < round(matrix.shape[0]*max_conflict):
                for ri in need:
                    for s_ci in s_group:
                        if col_list[s_ci][1][ri] >= 0:
                            col_list[s_ci][1][ri] = -1
                            pruned.append([s_ci,ri])
                            break
            
            for s_ci in s_list:
                if s_ci not in used and s_ci != d_ci and s_ci not in s_group:
                    #print(">>> ",col_list[s_ci])
                    match = []
                    count = 0
                    zeros = []
                    for ri,value in enumerate(col_list[s_ci][1]):
                        if value < 0:
                            zeros.append(ri)

                            if ri in need:
                                match.append(ri)
                                count += 1
                    
                    conflict = set(zeros)&set(slot)
                    if zeros:
                        if choice[1] < count:
                            choice[0] = s_ci
                            choice[1] = count
                            choice[2] = match
                            choice[3] = len(zeros)
                        
                        elif choice[1] == count and len(zeros) < choice[3]:
                            choice[0] = s_ci
                            choice[1] = count
                            choice[2] = match
                            choice[3] = len(zeros)

            
            if choice[0]:
                new_zero = list(need&set(zeros))
                for ri in choice[2]:
                    slot.append(ri)
                s_group.append(choice[0])
        
        #print("\n> ",col_list[d_ci])
        if s_group:
            used.append(d_ci)
            for s_ci in s_group:
                used.append(s_ci)
            #    print(">> ",col_list[s_ci])
            #print(">>> ",set(block)-set(slot))
        
        
        result.append([d_ci,s_group])
    
    packed_matrix = []
    not_used = [x for x in range(0,matrix.shape[0]) if x not in used]
    cols_mux = []
    for pack in result:
        sparse_cols = []
        if pack[1]:
            slot = []
            for i, ci in enumerate(pack[1]):
                sparse_cols.append(col_list[ci][1].tolist())
                cols_mux.append(group_len[pack[0]] + group_len[ci])
                for ri,value in enumerate(col_list[ci][1]):
                    if value < 0 and ri not in slot:
                        slot.append(ri)
            
            block = []
            for ri, value in enumerate(col_list[pack[0]][1]):
                if value >= 0:
                    block.append(ri)


            if set(block).issubset(slot):
                for ri, value in enumerate(col_list[pack[0]][1]):
                    for ci in range(0,len(sparse_cols)):
                        if sparse_cols[ci][ri] < 0:
                            sparse_cols[ci][ri] = value
                            break

            else:
                sparse_cols.append(col_list[pack[0]][1].tolist())

        #else:
        #    sparse_cols.append(col_list[pack[0]][1].tolist())
        #    cols_mux.append(group_len[pack[0]])
        
        for col in sparse_cols:
            packed_matrix.append(col)

    for ci in not_used:
        packed_matrix.append(col_list[ci][1])
        cols_mux.append(group_len[ci])
    
    return torch.tensor(packed_matrix), result, not_used, pruned,cols_mux

def pruned_column_scatter_v2(matrix, max_cols, max_conflict, group_len):
    col_list = [[ci,col] for ci, col in enumerate(matrix)]
    col_nonz = []
    for ci, col in col_list:
        count = 0
        for value in col:
            if value >= 0:
                count += 1

        col_nonz.append(count)
   
    d_list = [x[0] for n, x in sorted(zip(col_nonz, col_list),reverse=True) if n > 0]
    s_list = [x[0] for n, x in sorted(zip(col_nonz, col_list)) if n > 0]
    
    used = []

    result = []
    pruned = []
    for d_ci in d_list:
        block = []
        count = 0
        for ri, value in enumerate(col_list[d_ci][1]):
            if value >= 0:
                count += 1
                block.append(ri)
        
        if not block:
            continue
        if d_ci in used:
            continue

        slot = []
        
        s_group = []
        for l in range(0,max_cols):
            choice = [None, 0, [], 0]
            need= set(block)-set(slot)
            if not need:
                break
            
            elif len(need) < round(matrix.shape[0]*max_conflict):
                for ri in need:
                    for s_ci in s_group:
                        if col_list[s_ci][1][ri] >= 0:
                            col_list[s_ci][1][ri] = -1
                            pruned.append([s_ci,ri])
                            break
            
            for s_ci in s_list:
                if s_ci not in used and s_ci != d_ci and s_ci not in s_group:
                    #print(">>> ",col_list[s_ci])
                    match = []
                    count = 0
                    zeros = []
                    for ri,value in enumerate(col_list[s_ci][1]):
                        if value < 0:
                            zeros.append(ri)

                            if ri in need:
                                match.append(ri)
                                count += 1
                    
                    conflict = set(zeros)&set(slot)
                    if zeros:
                        if choice[1] < count:
                            choice[0] = s_ci
                            choice[1] = count
                            choice[2] = match
                            choice[3] = len(zeros)
                        
                        elif choice[1] == count and len(zeros) < choice[3]:
                            choice[0] = s_ci
                            choice[1] = count
                            choice[2] = match
                            choice[3] = len(zeros)

            
            if choice[0]:
                new_zero = list(need&set(zeros))
                for ri in choice[2]:
                    slot.append(ri)
                s_group.append(choice[0])
        
        #print("\n> ",col_list[d_ci])
        if s_group:
            used.append(d_ci)
            for s_ci in s_group:
                used.append(s_ci)
            #    print(">> ",col_list[s_ci])
            #print(">>> ",set(block)-set(slot))
        
        else:
            s_group.append(-1)
            #print("Broken")
        
        result.append([d_ci,s_group])
    
    packed_matrix = []
    not_used = [x for x in range(0,matrix.shape[0]) if x not in used]
    cols_mux = []
    for pack in result:
        sparse_cols = []

        if pack[1]:
            for i, ci in enumerate(pack[1]):
                sparse_cols.append(col_list[ci][1].tolist())
                cols_mux.append(group_len[pack[0]] + group_len[ci])


            for ri, value in enumerate(col_list[pack[0]][1]):
                for ci in range(0,len(sparse_cols)):
                    if sparse_cols[ci][ri] < 0:
                        sparse_cols[ci][ri] = value

        else:
            sparse_cols.append(col_list[pack[0]][1].tolist())
            cols_mux.append(group_len[pack[0]])
        
        for col in sparse_cols:
            packed_matrix.append(col)

    for ci in not_used:
        packed_matrix.append(col_list[ci][1])
        cols_mux.append(group_len[ci])
    
    return torch.tensor(packed_matrix), result, not_used, pruned,cols_mux

"""
print(pruned_tiles[0].transpose(0,1),"\n\n")

combined_matrix, groups_len = column_combine(pruned_tiles[0],0,mux_size)
sparsity = 0
for col in combined_matrix:
    for value in col:
        if value < 0:
            sparsity += 1

sparsity = sparsity/combined_matrix.shape[0]/combined_matrix.shape[1]*100
#print(combined_matrix.transpose(0,1))
print(groups_len)
print(round(sparsity,2),"\n\n")

half_over = [i for i,x in enumerate(groups_len) if x > (mux_size/2)]
half_over_mux = [x for i,x in enumerate(groups_len) if x > (mux_size/2)]
half_less = [i for i,x in enumerate(groups_len) if x <= (mux_size/2)]
print(mux_size/2)
print(half_over)
print(half_less)

look = combined_matrix.transpose(0,1)

half_over_matrix = look[half_over]
half_less_matrix = look[half_less]

#print(half_over_matrix,"\n\n")
#print(half_less_matrix)

balanced_matrix,_,_ = balancing(half_less_matrix.transpose(0,1))
#print(balanced_matrix.transpose(0,1),"\n\n")

packed_matrix, result, not_used, pruned, cols_mux = pruned_column_scatter(balanced_matrix.transpose(0,1),3,0,groups_len)

packed_matrix = torch.cat((half_over_matrix, packed_matrix),0)
#print(packed_matrix)
#print(result,not_used, pruned)
print(half_over_mux+cols_mux)

sparsity = 0
for col in packed_matrix:
    for value in col:
        if value < 0:
            sparsity += 1

sparsity = sparsity/packed_matrix.shape[0]/packed_matrix.shape[1]*100
print(round(sparsity,2))
"""
nonzeros = 0
for col in pruned_tiles[0]:
    for value in col:
        if value >= 0:
            nonzeros += 1

print(nonzeros)

###Test Case 1###
combined_matrix, groups_len = column_combine(pruned_tiles[0],0,mux_size/2)
#print(combined_matrix.transpose(0,1))
balanced_matrix,_,_ = row_balancing(combined_matrix)#.transpose(0,1))
#print(balanced_matrix.transpose(0,1))
packed_matrix, R, _, _, _ = pruned_column_scatter(balanced_matrix.transpose(0,1),3,0,groups_len)
nonzero, zero = counting(packed_matrix)
sparsity = zero/(nonzero+ zero)*100
print("\n",packed_matrix)
print("Test Case 1: ",round(sparsity,2), "with",nonzero)

###Test Case 2###
combined_matrix, groups_len = column_combine(pruned_tiles[0],0,mux_size)

half_over = [i for i,x in enumerate(groups_len) if x > (mux_size/2)]
half_less = [i for i,x in enumerate(groups_len) if x <= (mux_size/2)]
look = combined_matrix.transpose(0,1)

half_over_matrix = look[half_over]
half_less_matrix = look[half_less]

balanced_matrix,_,_ = row_balancing(half_less_matrix.transpose(0,1))

packed_matrix, R, _, _, _ = pruned_column_scatter(balanced_matrix.transpose(0,1),3,0,groups_len)
packed_matrix = torch.concat((look[half_over],packed_matrix),0)
nonzero, zero = counting(packed_matrix)

sparsity = zero/(nonzero+zero)*100
print("\n",packed_matrix)
print("Test Case 2: ",round(sparsity,2),"with",nonzero)


