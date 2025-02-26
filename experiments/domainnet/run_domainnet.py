import torch
import torchvision
from torchvision import transforms
from train_domainnet import CustomImageFolder
from smp_algos import get_input_sharing_layers, capture_inputs
import torchmetrics
import wandb
from compiler import get_permutation_spec
from checkpoints import load_model
from datasets_src import ImageNet
from activation_matching import activation_matching
from weight_matching import weight_matching
from partial_matching import partial_merge, qp_ratios, get_blocks
from copy import deepcopy, copy
from utils import Axis, count_linear_flops
import numpy as np
import os
import argparse

import gc

WANDB = True
MODEL_PATH = "/gscratch/sewoong/anasery/rebasin_merging/git-re-basin-fx/saved_models/"
model1 = load_model("rn50", "v1a").cuda()
model2 = load_model("rn50", "v1b").cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

# Training transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
test_loaders = dict([(x, torch.utils.data.DataLoader(CustomImageFolder(
    f"/scr/{x}/", f"/scr/{x}/test_images.txt", split_ratio=1.0, inmemory=False, transform=val_transforms), batch_size=256)) for x in ["clipart", "painting", "infograph", "real"]])

try:
    job_id = os.environ['SLURM_ARRAY_TASK_ID']
except:
    job_id = 1
assert job_id is not None
RESULT_FILE = "domainnet_results/smp_results_domainnet.json"
kidx = 0


def get_zip_ratios(initial_ratios, budget_ratio):
    layer_dict = {1.0: 4, 1.1: 3, 1.76: 2, 1.89: 1, 2.0: 0}
    new_ratios = {}
    for k,v in initial_ratios.items():
        if k.startswith('layer'):
            layernum = int(k.split('.')[0].split('layer')[1])
            print(k, layernum)
            if layernum <= layer_dict[budget_ratio]:
                new_ratios[k] = 0.
            else:
                new_ratios[k] = 1.
        else:
            new_ratios[k] = 0.
    return new_ratios


spec = get_permutation_spec(model1, ((5, 3, 224, 224),))
orig_ratios = {
    'conv1.weight': 0.0,
    'layer1.0.conv1.weight': 0.0,
    'layer1.0.conv2.weight': 0.0,
    'layer1.0.conv3.weight': 0.0,
    'layer1.1.conv1.weight': 0.0,
    'layer1.1.conv2.weight': 0.0,
    'layer1.1.conv3.weight': 0.0,
    'layer1.2.conv1.weight': 0.0,
    'layer1.2.conv2.weight': 0.0,
    'layer1.2.conv3.weight': 0.0,
    'layer2.0.conv1.weight': 0.0,
    'layer2.0.conv2.weight': 0.0,
    'layer2.0.conv3.weight': 0.0,
    'layer2.1.conv1.weight': 0.0,
    'layer2.1.conv2.weight': 0.0,
    'layer2.1.conv3.weight': 0.0,
    'layer2.2.conv1.weight': 0.0,
    'layer2.2.conv2.weight': 0.0,
    'layer2.2.conv3.weight': 0.0,
    'layer2.3.conv1.weight': 0.0,
    'layer2.3.conv2.weight': 0.0,
    'layer2.3.conv3.weight': 0.0,
    'layer3.0.conv1.weight': 0.0,
    'layer3.0.conv2.weight': 0.0,
    'layer3.0.conv3.weight': 0.0,
    'layer3.1.conv1.weight': 0.0,
    'layer3.1.conv2.weight': 0.0,
    'layer3.1.conv3.weight': 0.0,
    'layer3.2.conv1.weight': 0.0,
    'layer3.2.conv2.weight': 0.0,
    'layer3.2.conv3.weight': 0.0,
    'layer3.3.conv1.weight': 0.0,
    'layer3.3.conv2.weight': 0.0,
    'layer3.3.conv3.weight': 0.0,
    'layer3.4.conv1.weight': 0.0,
    'layer3.4.conv2.weight': 0.0,
    'layer3.4.conv3.weight': 0.0,
    'layer3.5.conv1.weight': 0.0,
    'layer3.5.conv2.weight': 0.0,
    'layer3.5.conv3.weight': 0.0,
    'layer3.6.conv1.weight': 0.0,
    'layer3.6.conv2.weight': 0.0,
    'layer3.6.conv3.weight': 0.0,
    'layer3.7.conv1.weight': 0.0,
    'layer3.7.conv2.weight': 0.0,
    'layer3.7.conv3.weight': 0.0,
    'layer3.8.conv1.weight': 0.0,
    'layer3.8.conv2.weight': 0.0,
    'layer3.8.conv3.weight': 0.0,
    'layer3.9.conv1.weight': 0.0,
    'layer3.9.conv2.weight': 0.0,
    'layer3.9.conv3.weight': 0.0,
    'layer3.10.conv1.weight': 0.0,
    'layer3.10.conv2.weight': 0.0,
    'layer3.10.conv3.weight': 0.0,
    'layer3.11.conv1.weight': 0.0,
    'layer3.11.conv2.weight': 0.0,
    'layer3.11.conv3.weight': 0.0,
    'layer3.12.conv1.weight': 0.0,
    'layer3.12.conv2.weight': 0.0,
    'layer3.12.conv3.weight': 0.0,
    'layer3.13.conv1.weight': 0.0,
    'layer3.13.conv2.weight': 0.0,
    'layer3.13.conv3.weight': 0.0,
    'layer3.14.conv1.weight': 0.0,
    'layer3.14.conv2.weight': 0.0,
    'layer3.14.conv3.weight': 0.0,
    'layer3.15.conv1.weight': 0.0,
    'layer3.15.conv2.weight': 0.0,
    'layer3.15.conv3.weight': 0.0,
    'layer3.16.conv1.weight': 0.0,
    'layer3.16.conv2.weight': 0.0,
    'layer3.16.conv3.weight': 0.0,
    'layer3.17.conv1.weight': 0.0,
    'layer3.17.conv2.weight': 0.0,
    'layer3.17.conv3.weight': 0.0,
    'layer3.18.conv1.weight': 0.0,
    'layer3.18.conv2.weight': 0.0,
    'layer3.18.conv3.weight': 0.0,
    'layer3.19.conv1.weight': 0.0,
    'layer3.19.conv2.weight': 0.0,
    'layer3.19.conv3.weight': 0.0,
    'layer3.20.conv1.weight': 0.0,
    'layer3.20.conv2.weight': 0.0,
    'layer3.20.conv3.weight': 0.0,
    'layer3.21.conv1.weight': 0.0,
    'layer3.21.conv2.weight': 0.0,
    'layer3.21.conv3.weight': 0.0,
    'layer3.22.conv1.weight': 0.0,
    'layer3.22.conv2.weight': 0.0,
    'layer3.22.conv3.weight': 0.0,
    'layer4.0.conv1.weight': 0.0,
    'layer4.0.conv2.weight': 0.0,
    'layer4.0.conv3.weight': 0.0,
    'layer4.1.conv1.weight': 0.0,
    'layer4.1.conv2.weight': 0.0,
    'layer4.1.conv3.weight': 0.0,
    'layer4.2.conv1.weight': 0.0,
    'layer4.2.conv2.weight': 0.0,
    'layer4.2.conv3.weight': 0.0,
}

job_id = 0
hyak_idx = 0
try:
    job_id = os.environ['SLURM_ARRAY_TASK_ID']
    
except:
    job_id = 1
    
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=1)
args = parser.parse_args()

def get_gradient_mask(perm_blocks, model_weights):
    masks = []
    for layer_name, params in model_weights.items():
        input_perms = perm_blocks[Axis(f"{layer_name}.weight", 1)]
        bi1, bi2, bi1c, bi2c = input_perms
        
        try:
            output_perms = perm_blocks[Axis(f"{layer_name}.weight", 0)]
            bo1, bo2, bo1c, bo2c = output_perms
        except KeyError:
            bo1, bo2, bo1c, bo2c = torch.arange(1000), torch.arange(1000), [], []

        ni, mi = len(bi1), len(bi1c)
        no, mo = len(bo1), len(bo1c)
        si12, si1, si2 = (
            slice(0, ni),
            slice(ni, ni + mi),
            slice(ni + mi, ni + 2 * mi),
        )
        so12, so1, so2 = (
            slice(0, no),
            slice(no, no + mo),
            slice(no + mo, no + 2 * mo),
        )
        
        for v in params.parameters():
            mask = torch.ones_like(v)

            if len(v.shape) >= 2:
                
                # These weights take input 1 to output 2 and vice-versa, so we make them non-trainable
                mask[si1,so2] = 0.
                mask[si2,so1] = 0.
            masks.append(mask)
    return masks


for model_type_pairs in [['rn101_v2a', 'rn101_v2b'], ['rn101_v2b', 'rn101_v2a']]:
    for dataset_pairs in [['clipart', 'painting'], ['clipart', 'infograph'], ['clipart', 'real'], ['painting', 'infograph'], ['painting', 'real'], ['infograph', 'real']]:    

            
        d1, d2 = dataset_pairs
        m1, m2 = model_type_pairs
        
        model1 = torchvision.models.resnet101(pretrained=False).cuda()
        model1.fc = torch.nn.Linear(model1.fc.in_features, 345).cuda()
        model2 = torchvision.models.resnet101(pretrained=False).cuda()
        model2.fc = torch.nn.Linear(model1.fc.in_features, 345).cuda()
        model1 = model1.cuda()
        model2 = model2.cuda()
        spec = get_permutation_spec(model1.float(), ((5, 3, 224, 224),))
        
        try:
            model1.load_state_dict(torch.load(
                f'/gscratch/sewoong/anasery/rebasin_merging/git-re-basin-fx/saved_models/domainnet_new/scr/{d1}/{m1}/model_epoch_30.pth'))
            model2.load_state_dict(torch.load(
                f'/gscratch/sewoong/anasery/rebasin_merging/git-re-basin-fx/saved_models/domainnet_new/scr/{d2}/{m2}/model_epoch_30.pth'))
        except:
            print("Model not found")
            continue
        try:
            del train_loader1
            del train_loader2
            gc.collect()
        except:
            pass
        
        for grb_data in ['orig']:
            
            if grb_data == 'orig':
                train_loader1 = CustomImageFolder(
                    f"/scr/{d1}/", f"/scr/{d1}/train_images.txt", inmemory=False, transform=train_transforms)
                train_loader2 = CustomImageFolder(
                    f"/scr/{d2}/", f"/scr/{d2}/train_images.txt", inmemory=False, transform=train_transforms)

                train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(
                    [train_loader1, train_loader2]), batch_size=16, shuffle=True, num_workers=2)
            else:
                train_loader = ImageNet().train_loader


            # budget_ratios = {Axis(k, 0): v for k, v in orig_ratios.items()}



            # merging = 'mean_eval_only'
            for merging in ['perm_gd', 'weight_matching']:
                for budget_ratio in [1.0, 1.1, 1.76, 1.9]:
                    
                    hyak_idx += 1
                    if hyak_idx != int(job_id):
                        continue
                    os.makedirs('/scr/saved_models/perm_gd/DomainNet/', exist_ok=True)
                    if 'weight_match' in merging:
                        perm_imnet, costs_imnet = weight_matching(spec=spec, state_as=model1.state_dict(), state_bs=model2.state_dict(), max_iter=100, verbose=True, seed=0, return_costs=True)
                    else:
                        perm_imnet, costs_imnet = activation_matching(
                            spec,
                            model1,
                            model2,
                            train_loader,
                            100,
                            output_costs=True,
                        )            
                    # R0 = np.load(
                    #     "/gscratch/sewoong/jhayase/oh/git-re-basin/git-re-basin-fx/archive/rn50-layerwise-0.npy")
                    # R1 = np.load(
                    #     "/gscratch/sewoong/jhayase/oh/git-re-basin/git-re-basin-fx/archive/rn50-layerwise-1.npy")

                    _, terms = count_linear_flops(spec, model1, ((1, 3, 224, 224),))

                    
                    # obj_weights = dict(
                        
                    #     zip(spec, (R1 - 0.4684)*(2 - budget_ratio) - (R0 - 0.7743)*(budget_ratio - 1)))


                    # budget_ratios = qp_ratios(spec, terms, budget_ratio, obj_weights)
                    budget_ratios = get_zip_ratios(orig_ratios, budget_ratio)
                    budget_ratios = {Axis(k, 0): v for k, v in budget_ratios.items()}
                    
                    new_perm, new_costs = {} , {}
                    if 'perm' in merging:
                        new_perm, new_costs = perm_imnet, costs_imnet
                    elif 'reg_mean' in merging or 'mean' in merging:
                        for k, v in perm_imnet.items():
                            new_perm[k] = torch.arange(len(v)).cuda()
                            new_costs[k] = torch.rand(costs_imnet[k].size(), dtype=costs_imnet[k].dtype).cuda()
                    else:
                        new_costs = costs_imnet
                        new_perm = perm_imnet
                    
                    if 'reg_mean' in merging:
                        model3 = deepcopy(model1)
                    else:                    
                        model3 = partial_merge(spec, model1, model2, new_perm, new_costs, budget_ratios)
                    model3.cuda()

                    if WANDB:
                        run = wandb.init(project="perm_gd_domainnet_rn101", config={"budget_ratio": budget_ratio, "m1": m1, "m2": m2, "d1": d1, "d2": d2, "merging": f"{merging}_eval_only", 'grb_data': grb_data}, reinit=True)      
                    model3.train()
                    for m in model3.modules():
                        if isinstance(m, torch.nn.BatchNorm2d):
                            m.reset_running_stats()
                    
                    bnidx = 0
                    for batch in train_loader:
                        with torch.no_grad():
                            x = batch[0].cuda().float()
                            _ = model3(x)
                        bnidx += 1
                        if bnidx > 100: break
                    model3.eval()
                    kaccs = {}
                    print("Evaluating the model")
                    for dset, test_loader in test_loaders.items():
                        acc1 = torchmetrics.Accuracy(
                            task="multiclass", num_classes=345).cuda()
                        acc5 = torchmetrics.Accuracy(
                            task="multiclass", num_classes=345, top_k=5).cuda()
                        sum_loss = 0
                        acc1.reset()
                        for images, labels in (test_loader):
                            with torch.no_grad():
                                x = images.cuda()
                                pred = model3(x)
                                acc1(pred, labels.cuda())
                        print(f"dset - {dset}, Final Accuracy - {acc1.compute().item()}")
                        kaccs[f"{dset}-accuracy"] = acc1.compute().item()
                    kaccs["budget_ratio"] = budget_ratio
                    run.log(kaccs)


                    activations_model_1 = {}
                    activations_model_2 = {}
                    model_layers = []
                    for name, module in model1.named_modules():
                        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                            model_layers.append(name)
                    inverted_ip_dict, new_ip_dict = get_input_sharing_layers(spec, model_layers)
                    model1_hook_handles = capture_inputs(model1, inverted_ip_dict, activations_model_1)
                    model2_hook_handles = capture_inputs(model2, inverted_ip_dict, activations_model_2)     

                    final_ip_dict = {}

                    def get_attr(obj, names):
                        # print(obj)
                        if len(names) == 1:
                            return getattr(obj, names[0])
                        else:
                            return get_attr(getattr(obj, names[0]), names[1:])


                    blocks = get_blocks(spec, perm_imnet, costs_imnet, budget_ratios, False)
                    perm_blocks = copy(blocks) 
                    for axis, pg in spec.items():
                        for ax in pg.state:
                            perm_blocks[ax] = perm_blocks[axis]

                    # blocks contained for each layer, which
                    axes_by_tensor = {}
                    for pg in spec.values():
                        for ax in pg.state:
                            axes_by_tensor.setdefault(ax.key, set()).add(ax.axis)



                    for k,v in new_ip_dict.items():
                        final_ip_dict[k] = sorted(list(v), key=lambda x: model_layers.index(x))
                    model3_dict = {}

                    for layer_to_test, v in final_ip_dict.items():
                        for lidx, l in enumerate(v):
                            l3 = get_attr(model3, l.strip(".weight").split('.')).float()
                            model3_dict[l.strip(".weight")] = deepcopy(l3).float()
                            model3_dict[l.strip(".weight")].requires_grad = True
                            for p in model3_dict[l.strip(".weight")].parameters():
                                p.requires_grad = True

                    gradient_masks = get_gradient_mask(perm_blocks, model3_dict)  # Makes zero weights non-trainable

                    m3_params = [x.parameters() for x in model3_dict.values()]
                    m3_params = [x for y in m3_params for x in y]
                    def get_model_orig_activations(model1, model2, perm_blocks, layer_group_name, layer_idx=0, layer_name=None):
                        
                        
                        if layer_name is None:
                            layer_name = layer_group_name
                        
                        l1 = get_attr(model1, layer_name.split('.'))
                        l2 = get_attr(model2, layer_name.split('.'))

                        input_perms = perm_blocks[Axis(f"{layer_name}.weight", 1)]
                        bi1, bi2, bi1c, bi2c = input_perms
                        
                        try:
                            output_perms = perm_blocks[Axis(f"{layer_name}.weight", 0)]
                            bo1, bo2, bo1c, bo2c = output_perms
                        except KeyError:
                            bo1, bo2, bo1c, bo2c = torch.arange(345), torch.arange(345), [], []
                        
                        ip1, ip2 = activations_model_1[layer_group_name][layer_idx], activations_model_2[layer_group_name][layer_idx]

                        acts_op_model1 = l1(activations_model_1[layer_group_name][layer_idx])
                        acts_op_model2 = l2(activations_model_2[layer_group_name][layer_idx])
                        if 'reg_mean' in merging:
                            acts_ip_stacked = torch.cat([ip1, ip2], dim=0)
                            acts_op_stacked = torch.cat([acts_op_model1, acts_op_model2], dim=0)
                        else:                        
                            acts_ip_stacked = torch.cat([(ip1[:,bi1]+ip2[:,bi2])/2, ip1[:,bi1c], ip2[:,bi2c]], dim=1)
                                
                            
                            acts_op_stacked = torch.cat([(acts_op_model1[:,bo1]+acts_op_model2[:,bo2])/2, acts_op_model1[:,bo1c], acts_op_model2[:,bo2c]], dim=1)
                        
                        # print(layer_group_name, layer_name, acts_ip_stacked.shape, acts_op_stacked.shape, len(bi1), len(bi2), len(bo1), len(bo2))
                        

                        
                        return acts_ip_stacked, acts_op_stacked

                    MAX_STEPS = 1 if 'eval_only' in merging else 500
                    optimizer = torch.optim.Adam(m3_params, lr=5e-3)
                    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEPS)
                    all_layer_loss = {}


                    if WANDB:
                        run = wandb.init(project="perm_gd_domainnet_rn101", config={"budget_ratio": budget_ratio, "m1": m1, "m2": m2, "d1": d1, "d2": d2, "merging": merging, 'grb_data': grb_data}, reinit=True)      

                    try:
                        for idx, batch in enumerate(train_loader):
                            x, y = batch
                            x = x.cuda().float()
                            with torch.no_grad():
                                model1(x)
                                model2(x)
                            if idx >= MAX_STEPS-1: break
                            total_loss = 0.
                            tracking_loss = 0.
                            for layer_to_test, v in final_ip_dict.items():
                                # print(v)
                                for lidx, l in enumerate(v):
                                    acts_ip, acts_op_stacked = get_model_orig_activations(model1, model2, perm_blocks, layer_to_test.strip(".weight"), lidx, l.strip(".weight"))
                                    layer_loss = 0.
                                    test_op = model3_dict[l.strip(".weight")](acts_ip)
                                    layer_loss = layer_loss + ((test_op-acts_op_stacked)**2).mean()
                                    
                                    if l.strip(".weight") not in all_layer_loss: all_layer_loss[l.strip(".weight")] = 0.
                                    total_loss += layer_loss
                                    all_layer_loss[l.strip(".weight")] += layer_loss
                            

                            optimizer.zero_grad()
                            total_loss.backward()
                            for param, mask in zip(m3_params, gradient_masks):
                                param.grad *= mask
                            optimizer.step()
                            lr_sched.step()
                            tracking_loss += total_loss.item()
                            keys = [x for x in activations_model_1.keys()]
                            for k in keys:
                                for l in activations_model_1[k]:
                                    del l
                                for l in activations_model_2[k]:
                                    del l
                                del activations_model_1[k]
                                del activations_model_2[k]        
                            if idx % 20 == 0 and idx:
                                if WANDB:
                                    wandb_metrics = {'step': idx, 'loss': tracking_loss / 20}
                                    print(f"Loss: {tracking_loss / 20:.3f}")
                                    tracking_loss = 0.
                                    for k in all_layer_loss.keys():
                                        wandb_metrics[f"loss_{k}"] = all_layer_loss[k] / 20
                                        all_layer_loss[k] = 0.
                                    run.log(wandb_metrics)

                            if idx % 200 == 0 and idx:
                                m3_sd = model3.state_dict()
                                for k,v in model3_dict.items():
                                    v_sd = v.state_dict()
                                    for k2 in v_sd.keys():
                                        m3_sd[f"{k}.{k2}"] = v_sd[k2]
                                        # print(f"Setting {k}.{k2} to {v_sd[k2].shape}")
                                torch.save(m3_sd, f"/scr/saved_models/perm_gd/DomainNet/{d1}-{d2}-{m1}-{m2}-perm_gd_budget-{budget_ratio}-budget-merging-{merging}.pt")
                                        
                                # torch.save(m3_sd, f"{MODEL_PATH}/perm_gd/perm_gd_budget-{budget_ratio}-budget.pt")
                    except torch.cuda.OutOfMemoryError:
                        print("OOM")
                        for obj in gc.get_objects():
                            try:
                                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                    print(type(obj), obj.size())
                            except:
                                pass           
                        raise Exception("OOM")  
                    
                    m3_sd = model3.state_dict()
                    for k,v in model3_dict.items():
                        v_sd = v.state_dict()
                        for k2 in v_sd.keys():
                            m3_sd[f"{k}.{k2}"] = v_sd[k2]
                            # print(f"Setting {k}.{k2} to {v_sd[k2].shape}")
                    torch.save(m3_sd, f"/scr/saved_models/perm_gd/DomainNet/{d1}-{d2}-{m1}-{m2}-perm_gd_budget-{budget_ratio}-budget-merging-{merging}.pt")            
                    # Reset batchnorm
                    model3 = model3.cuda().float()
                    model3.load_state_dict(torch.load(f"/scr/saved_models/perm_gd/DomainNet/{d1}-{d2}-{m1}-{m2}-perm_gd_budget-{budget_ratio}-budget-merging-{merging}.pt"))
                    model3.train()
                    for m in model3.modules():
                        if isinstance(m, torch.nn.BatchNorm2d):
                            m.reset_running_stats()
                    
                    bnidx = 0
                    for batch in train_loader:
                        with torch.no_grad():
                            x = batch[0].cuda().float()
                            _ = model3(x)
                            keys = [x for x in activations_model_1.keys()]
                            for k in keys:
                                for l in activations_model_1[k]:
                                    del l
                                for l in activations_model_2[k]:
                                    del l
                                del activations_model_1[k]
                                del activations_model_2[k]                     
                        bnidx += 1
                        if bnidx > 100: break
                    
                    # Evaluation
                    
                    model3.eval()
                    kaccs = {}
                    print("Evaluating the model")
                    for dset, test_loader in test_loaders.items():
                        acc1 = torchmetrics.Accuracy(
                            task="multiclass", num_classes=345).cuda()
                        acc5 = torchmetrics.Accuracy(
                            task="multiclass", num_classes=345, top_k=5).cuda()
                        sum_loss = 0
                        acc1.reset()
                        for images, labels in (test_loader):
                            with torch.no_grad():
                                x = images.cuda()
                                pred = model3(x)
                                acc1(pred, labels.cuda())
                                keys = [x for x in activations_model_1.keys()]
                                for k in keys:
                                    for l in activations_model_1[k]:
                                        del l
                                    for l in activations_model_2[k]:
                                        del l
                                    del activations_model_1[k]
                                    del activations_model_2[k]                         
                        print(f"dset - {dset}, Final Accuracy - {acc1.compute().item()}")
                        kaccs[f"{dset}-accuracy"] = acc1.compute().item()
                    kaccs["budget_ratio"] = budget_ratio
                    run.log(kaccs)
                #     del model3
                #     del m3_params
                # del model1
                # del model2
                    