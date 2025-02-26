import os
import torch
import torchvision
import torchmetrics
import wandb
import argparse

from tqdm import tqdm

from activation_matching import activation_matching
from weight_matching import weight_matching
from checkpoints import load_model
from utils import Axis, count_linear_flops
import numpy as np
import gc
from datasets_combined.interface import get_train_test_loaders
from src.datasets.common import get_dataloader
from src.modeling import ImageClassifier
from src.eval_smp import evaluate_with_model

import torchmetrics
import wandb
from compiler import get_permutation_spec
from activation_matching import activation_matching
from partial_matching import partial_merge, qp_ratios, get_blocks
from copy import deepcopy, copy
from utils import Axis, count_linear_flops


from perm_gd_generic import *

import torch
import torchvision
import torchmetrics
import wandb
from compiler import get_permutation_spec
from checkpoints import load_model
from datasets_src import ImageNet
from activation_matching import activation_matching
from partial_matching import partial_merge, qp_ratios, get_blocks
from copy import deepcopy, copy
from utils import Axis, count_linear_flops
import numpy as np
import os

import argparse

import gc

orig_ratios = {
    'conv1.weight': 0.0,
    'layer1.0.conv1.weight': 0.0,
    'layer1.0.conv2.weight': 0.0,
    'layer1.0.conv3.weight': 0.0,
    'layer1.1.conv1.weight': 0.0,
    'layer1.1.conv2.weight': 0.0,
    'layer1.2.conv1.weight': 0.0,
    'layer1.2.conv2.weight': 0.0,
    'layer2.0.conv1.weight': 0.0,
    'layer2.0.conv2.weight': 0.0,
    'layer2.0.conv3.weight': 0.0,
    'layer2.1.conv1.weight': 0.0,
    'layer2.1.conv2.weight': 0.0,
    'layer2.2.conv1.weight': 0.0,
    'layer2.2.conv2.weight': 0.0,
    'layer2.3.conv1.weight': 0.0,
    'layer2.3.conv2.weight': 0.0,
    'layer3.0.conv1.weight': 0.0,
    'layer3.0.conv2.weight': 0.0,
    'layer3.0.conv3.weight': 0.0,
    'layer3.1.conv1.weight': 0.0,
    'layer3.1.conv2.weight': 0.0,
    'layer3.2.conv1.weight': 0.0,
    'layer3.2.conv2.weight': 0.0,
    'layer3.3.conv1.weight': 0.0,
    'layer3.3.conv2.weight': 0.0,
    'layer3.4.conv1.weight': 0.0,
    'layer3.4.conv2.weight': 0.0,
    'layer3.5.conv1.weight': 0.0,
    'layer3.5.conv2.weight': 0.0,
    'layer4.0.conv1.weight': 0.0,
    'layer4.0.conv2.weight': 0.0,
    'layer4.0.conv3.weight': 0.0,
    'layer4.1.conv1.weight': 0.0,
    'layer4.1.conv2.weight': 0.0,
    'layer4.2.conv1.weight': 0.0,
    'layer4.2.conv2.weight': 0.0
}

# model_ckpt_paths = {'DTD': '/gscratch/scrubbed/anasery/GRB/models',
#                     'nabird': '/gscratch/scrubbed/anasery/GRB/models',
#                     'stanford_dogs': '/gscratch/scrubbed/anasery/GRB/models',
#                     'oxford_pets': '/gscratch/scrubbed/anasery/GRB/models'}

MODEL_PATH = "/gscratch/sewoong/anasery/rebasin_merging/fine_tuned_models/trained_models_tv/rn18"
DEBUG = False
# model_ckpts = {'cars': {False: '9a6667ffde88d30ffab428c7f64ab57ff1b08942d1204da5a68a7aa4e4fbab7b', True: '802bf438eae1415f51926bc4f5021d754bb3137475655977c4054de262760a80'}, 'cub': {False: 'c8cd69db71f3a0aa532f1543a4a87610ce770c491e94dc725c98721beac90112', True: 'cc38948df5c15ab51b6b83b44409a7b7880af7de42fbc36ffd9cee205e9ecbcd'}, 'dtd': {False: '5da02a32d6ff3a54d89c37fd88d24d994e0c1ea30a3d5c66ed420ea0b52bc14a', True: '55d1ff978de0cf87f6aa5699aac2e6731e658fdbd3d22c31f7960d7810ad302a'}, 'nabird': {False: '49cf1f2ac9b6fadfd47ddfc02d62921e7c155a3a3a43b89151d5b2b6ec84e074', True: 'ef6c18fc83b5c9e2417b86cad93e45a247cc464312488a59b741f5890f8fc95b'}, 'oxford_pets': {False: '9e5c3697a929a6f3d676f24e8823f62935852021b11e470e8f2a6deecd1f098a', True: '9b922f578d2a764575d9cfa7e1a8cf49a97bdda4f01a6750dec4001cd424530a'}, 'stanford_dogs': {False: '6dafd85a846a4be3b1d3ff5483e85443c88d373821afa5a0931ec9f730610f20', True: '29e6c0962cba8757d8e7c8328c81943dd181a875ff26f87d02d3b6e0838ac35e'}, 'sun397': {False: '7355f727cb936c809cc7c80da95eee165915cfe60db06c651c9667bf68fc005c', True: '750109388d9491a93eb3fcc3fb7fc6f7837d2235682bdbb5449e5bc3d2b7dbd4'}}
# model_ckpts = {'cars': {'false': '9a6667ffde88d30ffab428c7f64ab57ff1b08942d1204da5a68a7aa4e4fbab7b', 'true': '802bf438eae1415f51926bc4f5021d754bb3137475655977c4054de262760a80', 'v1a': 'd987a15410cd5d30ddf3000277365e2c7d3635acc9b480d3549e4dd7cec631ba', 'v1b': '9e4bd80da68871ee8d30116576806a6813a310038bac79f4aebf55133d84ecf9'}, 'cub': {'false': 'c8cd69db71f3a0aa532f1543a4a87610ce770c491e94dc725c98721beac90112', 'true': 'cc38948df5c15ab51b6b83b44409a7b7880af7de42fbc36ffd9cee205e9ecbcd', 'v1a': '4e6fa7b3b9a0e7f73398dec50aa57d77b5910f15b44688f490915407c22265b2', 'v1b': '60b5213e5f9d1a3c7ade8553cae4ad707a188dcc7f2ad11ce16cfff55bee0106'}, 'dtd': {'false': '5da02a32d6ff3a54d89c37fd88d24d994e0c1ea30a3d5c66ed420ea0b52bc14a', 'true': '55d1ff978de0cf87f6aa5699aac2e6731e658fdbd3d22c31f7960d7810ad302a', 'v1a': 'de2cc09c40430b50f7e521c870317894320c50b6d9aa0ea2a60e6d2398d65f7d', 'v1b': '678343fbd9baaad4099c1eab4f79090c1319b679700868565c55cf9ff0febff4'}, 'nabird': {'false': '49cf1f2ac9b6fadfd47ddfc02d62921e7c155a3a3a43b89151d5b2b6ec84e074', 'true': 'ef6c18fc83b5c9e2417b86cad93e45a247cc464312488a59b741f5890f8fc95b', 'v1a': 'ac591b5d773220bed9a9615d82a783584086a69f46df179e180c30bdadbe4ff5', 'v1b': '9603e49719e09336c8b401ef24c7edd078bc05cc9b381550855fa576249fc5c2'}, 'oxford_pets': {'false': '9e5c3697a929a6f3d676f24e8823f62935852021b11e470e8f2a6deecd1f098a', 'true': '9b922f578d2a764575d9cfa7e1a8cf49a97bdda4f01a6750dec4001cd424530a', 'v1a': '170d2ee2bd32b8ee155f0405077f27bf396eb802d64e7ec5a4b56e9ab9ae2adb', 'v1b': 'a66318e7373f4a4057f1711bf32759058615a1879b6fa8c74cd0e120d76b181b'}, 'stanford_dogs': {'false': '6dafd85a846a4be3b1d3ff5483e85443c88d373821afa5a0931ec9f730610f20', 'true': '29e6c0962cba8757d8e7c8328c81943dd181a875ff26f87d02d3b6e0838ac35e', 'v1a': 'b3e4d9be06e81d8106f9255fab0cc4672993e85dd3ec62b775976fc056fb8f84', 'v1b': '082df5203e50eba0b7da2aabaccbe3e17e73d453893f3488c8d4f71f039a67d1'}, 'sun397': {'false': '7355f727cb936c809cc7c80da95eee165915cfe60db06c651c9667bf68fc005c', 'true': '750109388d9491a93eb3fcc3fb7fc6f7837d2235682bdbb5449e5bc3d2b7dbd4', 'v1a': 'cc7701c32fc663c25b250c08d0c78bc5056dee298831fb425b9485c323895e0e', 'v1b': '3beb96156fe45cbc02122e3e5f39c85462972412fa83d978155d26b6c90ccca4'}}
# model_ckpts = {'cars': {'false': '9a6667ffde88d30ffab428c7f64ab57ff1b08942d1204da5a68a7aa4e4fbab7b', 
#                         'true': '802bf438eae1415f51926bc4f5021d754bb3137475655977c4054de262760a80', 
#                         'v1a': 'd987a15410cd5d30ddf3000277365e2c7d3635acc9b480d3549e4dd7cec631ba', 
#                         'v1b': '9e4bd80da68871ee8d30116576806a6813a310038bac79f4aebf55133d84ecf9'}, 
#                'cub': {'false': 'c8cd69db71f3a0aa532f1543a4a87610ce770c491e94dc725c98721beac90112', 
#                        'true': 'cc38948df5c15ab51b6b83b44409a7b7880af7de42fbc36ffd9cee205e9ecbcd', 
#                        'v1a': '4e6fa7b3b9a0e7f73398dec50aa57d77b5910f15b44688f490915407c22265b2', 
#                        'v1b': '60b5213e5f9d1a3c7ade8553cae4ad707a188dcc7f2ad11ce16cfff55bee0106'}, 
#                'dtd': {'false': '5da02a32d6ff3a54d89c37fd88d24d994e0c1ea30a3d5c66ed420ea0b52bc14a', 
#                        'true': '55d1ff978de0cf87f6aa5699aac2e6731e658fdbd3d22c31f7960d7810ad302a', 
#                        'v1a': 'de2cc09c40430b50f7e521c870317894320c50b6d9aa0ea2a60e6d2398d65f7d', 
#                        'v1b': '678343fbd9baaad4099c1eab4f79090c1319b679700868565c55cf9ff0febff4'},
#                'nabird': {'false': '49cf1f2ac9b6fadfd47ddfc02d62921e7c155a3a3a43b89151d5b2b6ec84e074', 
#                           'true': 'ef6c18fc83b5c9e2417b86cad93e45a247cc464312488a59b741f5890f8fc95b',
#                           'v1a': 'ac591b5d773220bed9a9615d82a783584086a69f46df179e180c30bdadbe4ff5',
#                           'v1b': '9603e49719e09336c8b401ef24c7edd078bc05cc9b381550855fa576249fc5c2'},
#                'oxford_pets': {'false': '9e5c3697a929a6f3d676f24e8823f62935852021b11e470e8f2a6deecd1f098a',
#                                'true': '9b922f578d2a764575d9cfa7e1a8cf49a97bdda4f01a6750dec4001cd424530a',
#                                'v1a': '170d2ee2bd32b8ee155f0405077f27bf396eb802d64e7ec5a4b56e9ab9ae2adb',
#                                'v1b': 'a66318e7373f4a4057f1711bf32759058615a1879b6fa8c74cd0e120d76b181b'},
#                'stanford_dogs': {'false': '6dafd85a846a4be3b1d3ff5483e85443c88d373821afa5a0931ec9f730610f20',
#                                  'true': '29e6c0962cba8757d8e7c8328c81943dd181a875ff26f87d02d3b6e0838ac35e', 
#                                  'v1a': 'b3e4d9be06e81d8106f9255fab0cc4672993e85dd3ec62b775976fc056fb8f84',
#                                  'v1b': '082df5203e50eba0b7da2aabaccbe3e17e73d453893f3488c8d4f71f039a67d1'},
#                'sun397': {'false': '7355f727cb936c809cc7c80da95eee165915cfe60db06c651c9667bf68fc005c', 
#                           'true': '750109388d9491a93eb3fcc3fb7fc6f7837d2235682bdbb5449e5bc3d2b7dbd4',
#                           'v1a': 'cc7701c32fc663c25b250c08d0c78bc5056dee298831fb425b9485c323895e0e',
#                           'v1b': '3beb96156fe45cbc02122e3e5f39c85462972412fa83d978155d26b6c90ccca4'}}

# model_ckpts = {'cub': {'v1a': 'b4dccc97bdd8f3c46f535aadac630bb88c31d47770b122ab01aa33a83bb81258', 'v1b': '6b1594f125853585657d2912480b6a132ae307902683357d0cc3332045f3845a'}, 'nabird': {'v1a': '4fa18a2662e2802da22e039e07b27e227cef8c7d0c136592919f7db85760c1ab', 'v1b': 'eccc4a2171c7f2bdf06c6d9f2a4bd24b74dd8f32836c9099f76aa4944de02781'}, 'oxford_pets': {'v1a': 'e697e497ce63c9b753a795dfc961f297d3696ab28d9f63afc3a5e68eb1e441d8', 'v1b': '4729867fee8e83715bfee034dea50e4e14c72fe65d3f807960d34e4c2b17ef71'}, 'stanford_dogs': {'v1a': '5b2dd733fa9153629492c6c2e362fc0df5f0a450d2c38b636f2a1f65ca502586', 'v1b': '41843362068f7447e6e5b5b8ca1eee05f7ac9a117efaf19f80509ba36e5c7277'}}
model_ckpts = {'cub': {'v1a': '3bcac5f8894367a1d5f1963b25734629ca656f8169312514d5083c78af0a565d', 'v1b': '1110d798ce054ddc05479cb65df545dd610f0f3d09b722db740d231e41a5c272'}, 'nabird': {'v1a': '0f2e90044eb921d172aa4746a67234ac1a848657529e4606292ae867760d7895', 'v1b': 'fa4ac3da8557743a0d1adb4b6c62c4adcbb38b852e683965264a975e690d494a'}, 'oxford_pets': {'v1a': '9530ed9bccc2c93a1a5603f3af1c070d4ee45cdb48686ed49a1f410a5aa8ca45', 'v1b': 'ab4839f3511998f634c127220579aed027896447b668b828bf9b3677ed822d26'}, 'stanford_dogs': {'v1a': '711186008d19ff8207823d3c994a2912cd52d712f05b40d1684a34ef8bd4bd98', 'v1b': 'b63dcd3d25235451b62473c506ca3cb26d4bea6d475194f0d20752190ac1ebbe'}}


def get_zip_ratios(initial_ratios, budget_ratio):
    layer_dict = {1.0: 4, 1.24: 3, 1.46: 2, 1.71: 1, 2.0: 0}
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


if __name__ == '__main__':
    datasets = ['cub', 'oxford_pets', 'stanford_dogs']
    # Enumerate all combinations of dataset pairs
    dataset_pairs = [(d1, d2) for i, d1 in enumerate(datasets)
                     for d2 in datasets[i+1:]]

    try:
        jobid = os.environ['SLURM_ARRAY_TASK_ID']
    except KeyError:
        jobid = 1

    idx = 0
    
    for seed in [1]:
        torch.manual_seed(seed)

        for pair in [
            # ['DTD', 'Cars'],  ['SUN397', 'Cars'], ['SUN397', 'DTD'],
                # ['cub', 'oxford_pets'], 
                # ['cub', 'stanford_dogs'],
                ['oxford_pets', 'nabird'],
                ['nabird', 'stanford_dogs'], 
                # ['cub', 'nabird'],
                # ['stanford_dogs', 'oxford_pets']
                ]:
                # ['nabird', 'stanford_dogs'], ['cub', 'nabird'], ['stanford_dogs', 'oxford_pets']]:
            # for pair in [['cub', 'oxford_pets']]:
            for pretrained in ['v1a', 'v1b']:
                # for merging in ['perm_gradmask', 'mean', 'mean_eval_only', 'perm_eval_only']:
                for grb_data  in ['orig']:
                    for merging in ['perm_gradmask']:
                        for budget_ratio in [1.0, 1.24, 1.46, 1.71]:
                            idx += 1
                            if idx != int(jobid):
                                continue
                            print(f"-"*20)
                            print(
                                f"Running {pair} with budget ratio {budget_ratio} on job {jobid}")
                            d1, d2 = pair
                            d1 = d1.lower()
                            d2 = d2.lower()
                            model1 = torchvision.models.resnet18().cuda()
                            model2 = torchvision.models.resnet18().cuda()
                            spec = get_permutation_spec(
                                model1, ((1, 3, 224, 224), ), verbose=False)

                            preprocess_fn = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                )
                            ])
                            print_every = 100
                            dataset_1_train, dataset_1_test = get_train_test_loaders(
                                d1, 16, True, directory_suffix=str(jobid))

                            dataset_2_train, dataset_2_test = get_train_test_loaders(
                                d2, 16, True, directory_suffix=str(jobid))
                            try:
                                num_classes_1 = len(dataset_1_train.dataset.classes)
                            except AttributeError:
                                num_classes_1 = np.max(
                                    dataset_1_train.dataset.targets) + 1

                            try:
                                num_classes_2 = len(dataset_2_train.dataset.classes)
                            except AttributeError:
                                num_classes_2 = np.max(
                                    dataset_2_train.dataset.targets) + 1

                            
                            
                            bn_train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([dataset_1_train.dataset, dataset_2_train.dataset]),
                                                                    batch_size=32, shuffle=True, num_workers=2)

                            if 'imagenet' in grb_data:
                                train_loader = ImageNet(batch_size=32).train_loader
                            else:
                                train_loader = bn_train_loader
                            
                            model1.fc = torch.nn.Linear(512, num_classes_1).cuda()
                            model2.fc = torch.nn.Linear(512, num_classes_2).cuda()

                            non_pt = 'v1b' if pretrained == 'v1a' else 'v1a'

                            model_dict_1 = torch.load(
                                f"{MODEL_PATH}/{model_ckpts[d1][pretrained]}.pth")
                            model_dict_2 = torch.load(
                                f"{MODEL_PATH}/{model_ckpts[d2][non_pt]}.pth")

                            model1.load_state_dict(model_dict_1['model'])
                            model2.load_state_dict(model_dict_2['model'])

                            fc = [model1.fc, model2.fc]
                            model1.fc = torch.nn.Identity()
                            model2.fc = torch.nn.Identity()

                            # Load linear classifiers separately

                            # Merge datasets
                            model1 = model1.cuda()
                            model2 = model2.cuda()

                            if 'weight' not in merging:
                                perm, costs = activation_matching(
                                    spec,
                                    model1,
                                    model2,
                                    train_loader,
                                    1 if DEBUG else 100,
                                    output_costs=True,
                                )
                            else:
                                perm, costs = weight_matching(spec=spec, state_as=model1.state_dict(), state_bs=model2.state_dict(), max_iter=100, verbose=False, seed=seed, return_costs=True)
                                
                            new_perm, new_costs = {}, {}
                            if 'perm' in merging:
                                new_perm, new_costs = perm, costs
                            elif 'reg_mean' in merging or 'mean' in merging:
                                for k, v in perm.items():
                                    new_perm[k] = torch.arange(len(v)).cuda()
                                    new_costs[k] = torch.rand(
                                        costs[k].size(), dtype=costs[k].dtype).cuda()
                            else:
                                new_perm, new_costs = perm, costs
                            
                            
                            perm = new_perm
                            costs = new_costs

                            R0 = np.load(
                                "/gscratch/sewoong/anasery/rebasin_merging/git-re-basin-fx/layer_wise_results/rn_18/R0.npy")
                            R1 = np.load(
                                "/gscratch/sewoong/anasery/rebasin_merging/git-re-basin-fx/layer_wise_results/rn_18/R1.npy")

                            _, terms = count_linear_flops(
                                spec, model1, ((1, 3, 224, 224),))

                            obj_weights = dict(
                                zip(spec, (R1 - 0.0169)*(2 - budget_ratio) - (R0 - 0.6647)*(budget_ratio - 1)))
                            budget_ratios = qp_ratios(
                                spec, terms, budget_ratio, obj_weights)

                            budget_ratios = get_zip_ratios(orig_ratios, budget_ratio)
                            
                            budget_ratios = {Axis(k, 0): v for k,
                                            v in budget_ratios.items()}
                            # budget_ratios = {Axis(k, 0): budget_ratio - 1.0 for k,
                            #                 v in budget_ratios.items()}                            
                            if 'reg_mean' in merging:
                                model3 = deepcopy(model1)
                            else:
                                model3 = partial_merge(
                                    spec, model1, model2, perm, costs, budget_ratios)
                            model3.cuda()
                            model3.train()
                            lp_train_loaders = [dataset_1_train, dataset_2_train]

                            #### Eval only  #####



                            wandb_run = wandb.init(project=f"perm_gd_tv_rn_18_zipsched_seed_{seed}", config={"budget_ratio": budget_ratio, 'merge type': f"{merging}-eval_only-actual",
                                                                                                'd1': d1, 'd2': d2, 'pretrained': pretrained, 'grb_data': f"{grb_data}-bn-orig-linearprobe"
                                                                                                }, reinit=True)
                            for bidx, batch in enumerate(train_loader):
                                model3(batch[0].cuda())
                                if bidx > 100:
                                    break
                            fc_perm = get_fc_perm(perm,spec, costs, budget_ratios )
                            # for didx, test_loader in enumerate([dataset_1_test, dataset_2_test]):
                            #     # acc = eval_perm_model(model3.cuda(), fc[didx].cuda(), test_loader,  int(fc[didx].out_features), fc_perm, didx)
                            #     # wandb_run.log({f'{pair[didx]}_acc': acc})

                            #     train_eval_linear_probe(model3, lp_train_loaders[didx], test_loader, fc[didx].out_features, wandb_run, pair[didx], lr=1e-3, epochs=50)
                            
                            
                            
                            ##### Actual stuff #####        
                            
                                    
                            wandb_run = wandb.init(project=f"perm_gd_tv_rn_18_zipsched_seed_{seed}", config={"budget_ratio": budget_ratio, 'merge type': f"{merging}-actual",
                                                                                                'd1': d1, 'd2': d2, 'pretrained': pretrained, 'grb_data': f"{grb_data}-bn-orig-linearprobe"
                                                                                                }, reinit=True)

                            if 'eval_only' not in merging:

                                m3_new = train(train_loader, model1, model2, model3, spec, perm, costs, budget_ratios, True, 1 if DEBUG else 400, wandb_run, True, merging, rn_18=True)
                            else:
                                m3_new = model3

                            m3_new.train()
                            # Recompute batch norm
                            for bidx, batch in enumerate(train_loader):
                                m3_new(batch[0].cuda())
                                if bidx > 100:
                                    break
                            # Eval m3_new
                            # fc_perm = get_fc_perm(perm,spec, costs, budget_ratios )
                            for didx, test_loader in enumerate([dataset_1_test, dataset_2_test]):
                                # dataloader = get_dataloader(dataset, is_train=False, image_encoder=None, args=None)

                                # acc = eval_perm_model(m3_new.cuda(), fc[didx].cuda(), test_loader,  int(fc[didx].out_features), fc_perm, didx)

                                # # Log to wandb
                                # wandb_run.log({f'{pair[didx]}_acc': acc})
                                train_eval_linear_probe(m3_new, lp_train_loaders[didx], test_loader, fc[didx].out_features, wandb_run, pair[didx], lr=1e-3, epochs=50)
