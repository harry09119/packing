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
    #if len(col_rows) > 1:
    for g, grp in enumerate(col_group):
        grp_rows = set(grp)
        before_density = len(grp_rows)
        after_density = len(col_rows | grp_rows)
        density_improve = after_density - before_density
        conflict = len(col_rows & grp_rows)
        #conflict = len(col_rows & grp_rows)

        if (conflict+col_group_conflict[g]) <= max_conflict:
            if len(col_group_index[g]) < (mux_size):
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
    print(before[:,c])

print("####Column Combine####")
after = torch.full((wgt.shape[0],len(col_group)),-1)
for g,cols in enumerate(col_group_index):
    for c,col in enumerate(cols):
        for r in cols_rows[col]:
            after[r,g] = col


for c in range(0,len(col_group)):
    print(col_group_conflict[c],after[:,c])

print("\n<<<RESULT>>>\n")
print("> Run Test at Weight ["+str(row_size)+"X",str(col_size),"] with mux:", mux_size,"and conflict: "+str(max_conflict/row_size*100)+"%")
print("\n> Before Column Combining, Density:"+str(density.item()*100)+"%")
print("> After Column Combining, Density: "+str(group_density)+"%")
print("> With Pruned Non_zero Percentage: "+str(round(pruned_nonzeros,3))+"% from "+str(sum(col_group_conflict))+"/"+str(full_nonzeros))

after_t = after.transpose(0,1)
counts = []
for g in range(0,len(after_t)):
    count = 0
    for r in range(0,len(after_t[g])):
        if after_t[g,r] > -1:
            count+=1
    counts.append(count)

sorted_after = [torch.tensor(x) for _,x in sorted(zip(counts,after_t.tolist()))]
counts = [i for i,_ in sorted(zip(counts,after_t.tolist()))]
row_counts = torch.zeros(1,row_size,dtype = int).squeeze()
slots = [row_size - i for i in counts]
for i, col in enumerate(sorted_after):
    print(col,counts[i],row_size-counts[i],i)
    for j,value in enumerate(col):
        if value > -1:
            row_counts[j] +=1

print(row_counts, "row counts")
devide = 0
for i in range(len(sorted_after)-1,-1,-1):
    long_values = sum(counts[i:])
    short_slots = sum(slots[:i])
    if short_slots <= long_values:
        devide = i
        break
print("\n",devide,"\n")

after_slot = []
after_value = []
for i in range(0,len(sorted_after)):
    slot = []
    value = []
    for j in range(0, row_size):
        if sorted_after[i][j] < 0:
            slot.append(j)
        else:
            value.append(j)

    after_slot.append(torch.tensor(slot))
    after_value.append(torch.tensor(value))

#for s,v in zip(after_slot,after_value):
#    print(s,v)

floor = []
tile = []
used = []

for c_low in range(len(sorted_after)-1,-1,-1):
    if c_low not in used:
        blocks = set(after_value[c_low].tolist())
        slots = set([])
        group = []
        used_tmp = []
        for l in range(0,limit):
            required_slots = blocks - slots
            if not (required_slots):
                break
            chosen = -1
            max_improve = 0
            min_conflict = len(slots)
            for c_high in range(0,len(sorted_after)):
                if c_high not in used and c_high != c_low:
                    new_col = set(after_slot[c_high].tolist())
                    conflict = (slots & new_col)
                    new_slots = new_col - conflict
                    improve = len(new_slots & required_slots)
                    if improve > max_improve:
                        chosen = c_high
                        max_improve = improve
                        min_conflict = len(conflict)

                    elif improve == max_improve:
                        if len(conflict) < min_conflict:
                            chosen = c_high
                            min_conflict = len(conflict)
            slots = slots | set(after_slot[chosen].tolist())
            if chosen >= 0:
                used_tmp.append(chosen)
                group.append(chosen)
                slots = slots | set(after_slot[chosen].tolist())
            else:
                used_tmp = []
                group = []
                break

        print(blocks.issubset(slots),blocks,"->",slots,"no:",blocks-slots,"\nwith:",c_low,used_tmp)
        if blocks.issubset(slots):
            used = list(set(used)|set(used_tmp))
            used.append(c_low)
            tile.append(c_low)
            floor.append(group)

print(floor)
print(tile)
used = set(tile)
for f in floor:
    used = used | set(f)
left = list(set(range(0,len(sorted_after)))-set(used))
print(left)

#check

packed_after = []
for t in range(0,len(tile)):
    used = []
    for f in range(0,len(floor[t])):
        col = sorted_after[floor[t][f]]
        for r in range(0, row_size):
            if sorted_after[tile[t]][r] > -1:
                if col[r] == -1:
                    if r not in used:
                        used.append(r)
                        col[r] = sorted_after[tile[t]][r]
        packed_after.append(col)

for l in left:
    packed_after.append(sorted_after[l])

for i,col in enumerate(packed_after):
    print(col,i)

count = 0
row_count = [0 for x in range(0,row_size)]
for c,col in enumerate(packed_after):
    for r,value in enumerate(col):
        if value > -1:
            count+=1
            row_count[(r)] +=1

print(len(packed_after))
print(count,row_size,len(packed_after))
print(row_count)
print((count/len(packed_after)/row_size)*100)
