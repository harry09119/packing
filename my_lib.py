import timm as tm
import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm 
import copy
import random
import math
from collections import Counter
#vit = tm.create_model('vit_base_patch16_224', pretrained=True)

##control##
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

    group_index = []
    group = []
    group_conflict = []
    for ci in range(0,matrix.shape[1]):
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

    group_len = [len(group) for group in group_index if group]

    packed_matrix = torch.full((len(group_len),matrix.shape[0]),-1)
    
    pruned = []
    for gi, cols in enumerate(group_index):
        for ci in cols:
            for ri in nonzero_i[ci]:
                if packed_matrix[gi,ri] >= 0:
                    pruned.append([ri,ci])
                packed_matrix[gi,ri] = ci
   
    group_len = [len(group) for group in group_index if group]
    """
    pruned = []
    for gi, cis in enumerate(group_index):
        kind_nz = []
        for ci in cis:
            for nz in nonzero_i[ci]:
                if nz not in kind_nz:
                    kind_nz.append(nz)
                else:
                    pruned.append([nz,ci])
    """
    return packed_matrix.transpose(0,1), group_len, pruned

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
    for o, d_ci in enumerate(d_list):
        block = []
        d_mux = group_len[d_ci]
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
        s_change = []
        for l in range(0,max_cols):
            choice = [-1, 0, [], 0]
            need= set(block)-set(slot)
            if not need:
                break
            if len(need) <= round(matrix.shape[1] * max_conflict) and len(block) > round(matrix.shape[1] * max_conflict) and len(s_group)>0:
                for ri in need:
                    col_list[d_ci][1][ri] = -1
                    pruned.append([ri,d_ci])
                    slot.append(ri)
                break

            not_used_ci = [s_ci for s_ci in s_list if s_ci not in used and s_ci != d_ci and s_ci not in s_group]
            not_used_cols = [copy.deepcopy(col_list[s_ci][1]) for s_ci in s_list if s_ci not in used and s_ci != d_ci and s_ci not in s_group]

            for i, s_ci in enumerate(not_used_ci):
                prev = -1
                #print("\nWith:",i, col_list[d_ci][1] )
                #print("> Before: ",not_used_cols[i])
                for ri in need:
                    if ri == 0 or not_used_cols[i][ri] < 0:
                        prev = ri
                        continue
                    
                    start = ri - 1
                    
                    while not_used_cols[i][start]>=0 and start > prev:
                        start -= 1
                    if start <= prev:
                        continue

                    for rj in range(start,ri):
                        not_used_cols[i][rj] = not_used_cols[i][rj+1]
                    
                    not_used_cols[i][ri] = -1
                #print("> After: ",not_used_cols[i])
            """
            if o == 0:
                print("\nDebugging")
                print(col_list[d_ci][1],"\n")
                for i, col in enumerate(not_used_cols):
                    print(col_list[not_used_ci[i]][1])
                    print("To")
                    print(col)
            """
            for i, s_ci in enumerate(not_used_ci):
                s_mux = group_len[s_ci]
                if s_mux > (max_cols - d_mux):
                    continue
                match = []
                count = 0
                zeros = []
                for ri,value in enumerate(not_used_cols[i]):
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

            if choice[0] >= 0:
                new_zero = list(need&set(zeros))
                for ri in choice[2]:
                    slot.append(ri)
                s_group.append(choice[0])
                s_change.append(not_used_cols[not_used_ci.index(choice[0])])
        
        #print("\n> ",col_list[d_ci])
        if s_group:
            if set(block).issubset(set(slot)):
                used.append(d_ci)
                for i, s_ci in enumerate(s_group):
                    col_list[s_ci][1] = s_change[i]
                    used.append(s_ci)
            else:
                s_group = []
                s_change = []
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
        packed_matrix.append(col_list[ci][1].tolist())
        cols_mux.append(group_len[ci])
    
    packed_matrix = torch.tensor(packed_matrix)
    return packed_matrix.transpose(0,1), pruned

