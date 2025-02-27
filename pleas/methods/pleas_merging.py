import torch
import torchmetrics
import tqdm.auto as tqdm
from copy import copy

from pleas.core.utils import Axis
from pleas.methods.partial_matching import get_blocks
from copy import deepcopy


def get_gradient_mask(perm_blocks, model_weights):
    """
    Generate binary masks to restrict gradient updates to appropriate weights.
    
    In PLeaS merging, only certain weights should be updated during training.
    This function creates binary masks to zero out gradients for weights that
    should remain fixed.
    
    Args:
        perm_blocks (dict): Dictionary of permutation blocks
        model_weights (dict): Dictionary of model layer weights
        
    Returns:
        list: List of binary masks matching the model parameters
    """
    masks = []
    for layer_name, params in model_weights.items():
        try:
            input_perms = perm_blocks[Axis(f"{layer_name}.weight", 1)]
            bi1, bi2, bi1c, bi2c = input_perms
        except KeyError:
            bi1, bi2, bi1c, bi2c = torch.arange(3), torch.arange(3), [], []
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


def get_model_orig_activations(model1, model2, perm_blocks, layer_name, activations_model_1, activations_model_2, ip_dim=1, separate_classifier=False, merging='perm_gradmask', num_classes=1000, model_type='rn50'):
    """
    Extract and process activations from two models for a given layer.
    
    This function extracts activations from both source models, applies the appropriate
    permutations, and formats them for use in the PLeaS optimization.
    
    Args:
        model1 (torch.nn.Module): First source model
        model2 (torch.nn.Module): Second source model
        perm_blocks (dict): Dictionary of permutation blocks
        layer_name (str): Name of the layer to process
        activations_model_1 (dict): Activations from the first model
        activations_model_2 (dict): Activations from the second model
        ip_dim (int, optional): Input dimension. Defaults to 1.
        separate_classifier (bool, optional): Whether to use separate classifier. Defaults to False.
        merging (str, optional): Merging method. Defaults to 'perm_gradmask'.
        num_classes (int, optional): Number of output classes. Defaults to 1000.
        rn_18 (bool/str, optional): Specific model type flag. Defaults to False.
        
    Returns:
        tuple: Stacked input and output activations for the merged model
    """
    l1 = get_attr(model1, layer_name.split('.'))
    l2 = get_attr(model2, layer_name.split('.'))

    ip1, ip2 = activations_model_1[layer_name], activations_model_2[layer_name]

    try:
        output_perms = perm_blocks[Axis(f"{layer_name}.weight", 0)]
        bo1, bo2, bo1c, bo2c = output_perms
    except KeyError:
        if separate_classifier:
            if model_type == 'rn50' or model_type == 'rn101':
                bo1, bo2, bo1c, bo2c = torch.arange(2048), torch.arange(2048), torch.tensor([]), torch.tensor([])
            elif model_type == 'rn20':
                bo1, bo2, bo1c, bo2c = torch.arange(1024), torch.arange(1024), torch.tensor([]), torch.tensor([])
            elif model_type == 'rn18':
                bo1, bo2, bo1c, bo2c = torch.arange(512), torch.arange(512), torch.tensor([]), torch.tensor([])
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            bo1, bo2, bo1c, bo2c = torch.arange(num_classes), torch.arange(num_classes), torch.tensor([]), torch.tensor([])
            
    try:
        input_perms = perm_blocks[Axis(f"{layer_name}.weight", 1)]
        bi1, bi2, bi1c, bi2c = input_perms
    except KeyError:
        bi1, bi2, bi1c, bi2c = torch.arange(ip1.shape[1]), torch.arange(ip1.shape[1]), torch.tensor([]), torch.tensor([])    

    acts_op_model1 = l1(activations_model_1[layer_name])
    acts_op_model2 = l2(activations_model_2[layer_name])

    i11 = ip1.index_select(ip_dim, bi1.int().to(ip1.device))
    i22 = ip2.index_select(ip_dim, bi2.int().to(ip1.device))
    i1c = ip1.index_select(ip_dim, bi1c.int().to(ip1.device))
    i2c = ip2.index_select(ip_dim, bi2c.int().to(ip1.device))
    o11 = acts_op_model1.index_select(ip_dim, bo1.int().to(ip1.device))
    o22 = acts_op_model2.index_select(ip_dim, bo2.int().to(ip1.device))
    o1c = acts_op_model1.index_select(ip_dim, bo1c.int().to(ip1.device))
    o2c = acts_op_model2.index_select(ip_dim, bo2c.int().to(ip1.device))
    
    if merging == 'reg_mean':
        # Assuming 1.0 only
        acts_ip_stacked = torch.cat([ip1, ip2], dim=0)
        acts_op_stacked = torch.cat([acts_op_model1, acts_op_model2], dim=0)
        
        return acts_ip_stacked, acts_op_stacked
    
    elif 'perm_separatels' in merging:
        acts_ip_stacked = torch.cat([torch.cat([(i11), i1c, torch.zeros_like(i2c)], dim=ip_dim),
                                        torch.cat([(i22), torch.zeros_like(i1c), i2c], dim=ip_dim)], dim=0)   
        acts_op_stacked = torch.cat([torch.cat([(o11), o1c, torch.zeros_like(o2c)], dim=1),
                                        torch.cat([(o22), torch.zeros_like(o1c), o2c], dim=1)], dim=0)
        return acts_ip_stacked, acts_op_stacked
                    
    elif 'perm_mixedls' in merging:
        acts_ip_stacked = torch.cat([torch.cat([(i11+i22)/2, i1c, torch.zeros_like(i2c)], dim=1),
                                        torch.cat([(i11+i22)/2, torch.zeros_like(i1c), i2c], dim=1)], dim=0)   
        acts_op_stacked = torch.cat([torch.cat([(o11), o1c, torch.zeros_like(o2c)], dim=1),
                                        torch.cat([(o22), torch.zeros_like(o1c), o2c], dim=1)], dim=0)
        return acts_ip_stacked, acts_op_stacked

    acts_ip_stacked = torch.cat([(i11+i22)/2, i1c, i2c], dim=ip_dim)
    acts_op_stacked = torch.cat([(o11+o22)/2, o1c, o2c], dim=ip_dim)
    
    return acts_ip_stacked, acts_op_stacked


def get_model_dict_and_params(model3):
    """
    Extract trainable layers and parameters from a model.
    
    This function creates a dictionary of trainable layers and a flat list
    of parameters for optimization.
    
    Args:
        model3 (torch.nn.Module): Model to extract parameters from
        
    Returns:
        tuple: (model, layer_dict, parameter_list)
    """
    model3_dict = {}

    for name, v in model3.named_modules():
        if isinstance(v, torch.nn.Conv2d) or isinstance(v, torch.nn.Linear):
            model3_dict[name] = deepcopy(v).float().cuda()
            model3_dict[name].requires_grad = True

            for p in model3_dict[name].parameters():
                p.requires_grad = True
    m3_params = [x.parameters() for x in model3_dict.values()]
    m3_params = [x for y in m3_params for x in y]
    
    return model3, model3_dict, m3_params


def get_attr(obj, names):
    """
    Access nested attributes in an object.
    
    Args:
        obj: Object to access attributes from
        names (list): List of attribute names to access in sequence
        
    Returns:
        The requested attribute
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def capture_input_hook(group_name, acts):
    """
    Create a hook function to capture inputs to a module.
    
    Args:
        group_name (str): Key to store the captured input
        acts (dict): Dictionary to store activations
        
    Returns:
        callable: Hook function
    """
    def hook(module, input, output):
        if isinstance(input, tuple):
            assert len(input) == 1
            input = input[0]
        acts[group_name] = input
    return hook


def capture_inputs(model, act_dict):
    """
    Register hooks to capture inputs to all relevant layers in a model.
    
    Args:
        model (torch.nn.Module): Model to register hooks on
        act_dict (dict): Dictionary to store activations
        
    Returns:
        list: List of hook handles
    """
    hook_handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.LayerNorm):
            hook_handles.append(module.register_forward_hook(capture_input_hook(name, act_dict) ))    
    return hook_handles


def step(batch, model1, model2, model3_dict, m3_params, optimizer, perm_blocks, all_layer_loss, activations_model_1, activations_model_2, grad_masks=None, separate_classifier=False, merging='perm_gradmask', num_classes=1000, verbose=False, model_type='rn50'):
    """
    Perform a single optimization step for PLeaS merging.
    
    This function performs forward passes through the source models,
    computes the loss for each layer in the merged model, and updates
    the parameters of the merged model to better match the activations
    of the source models.
    
    Args:
        batch (tuple): Input batch (x, y)
        model1 (torch.nn.Module): First source model
        model2 (torch.nn.Module): Second source model
        model3_dict (dict): Dictionary of trainable layers in the merged model
        m3_params (list): Flat list of parameters to optimize
        optimizer (torch.optim.Optimizer): Optimizer
        perm_blocks (dict): Dictionary of permutation blocks
        all_layer_loss (dict): Dictionary to accumulate losses per layer
        activations_model_1 (dict): Activations from the first model
        activations_model_2 (dict): Activations from the second model
        grad_masks (list, optional): Masks for gradient updates. Defaults to None.
        separate_classifier (bool, optional): Whether to use separate classifier. Defaults to False.
        merging (str, optional): Merging method. Defaults to 'perm_gradmask'.
        num_classes (int, optional): Number of output classes. Defaults to 1000.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        rn_18 (bool/str, optional): Specific model type flag. Defaults to False.
        
    Returns:
        torch.Tensor: Total loss for this step
    """
    x,_ = batch
    with torch.no_grad():
        x = x.cuda()
        model1(x)
        model2(x)
    optimizer.zero_grad()
    total_loss = 0.
    for layer_name, layer in model3_dict.items():
        try:
            acts_ip_stacked, acts_op_stacked = get_model_orig_activations(model1, model2, perm_blocks, layer_name, activations_model_1, activations_model_2, separate_classifier=separate_classifier, merging=merging, num_classes=num_classes, model_type=model_type)
        except KeyError:
            print(f"Key error on {layer_name}")
            continue
        acts_ip_stacked = acts_ip_stacked.cuda()
        acts_op_stacked = acts_op_stacked.cuda()

        if verbose: print(f"Working on {layer_name}, {layer}, {layer.weight.shape} shapes - ip: {acts_ip_stacked.shape}, op: {acts_op_stacked.shape}")
        acts_op_model3 = layer(acts_ip_stacked)
        layer_loss = ((acts_op_model3-acts_op_stacked)**2).mean()  # Calculate MSE loss
        total_loss += layer_loss
        all_layer_loss[layer_name] += layer_loss
    

    total_loss.backward()
    if grad_masks is not None:
        for param, mask in zip(m3_params, grad_masks):
            param.grad *= mask  # Apply gradient masks to enforce structural constraints
    optimizer.step()
    
    # Clean up activations to prevent memory leaks
    keys = [x for x in activations_model_1.keys()]
    for k in keys:
        for l in activations_model_1[k]:
            del l
        for l in activations_model_2[k]:
            del l
        del activations_model_1[k]
        del activations_model_2[k]        
    return total_loss


def train(dataloader, model1, model2, model3, spec, perm, costs, budget_ratios, WANDB, MAX_STEPS, wandb_run, separate_classifier=False, merging='perm_gradmask', num_classes=1000, lr=5e-4, verbose=False, model_type='rn50'):
    """
    Train a merged model using the PLeaS algorithm.
    
    This is the main training function for PLeaS merging, which optimizes the parameters
    of a merged model to better match the activations of two source models.
    
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for training data
        model1 (torch.nn.Module): First source model
        model2 (torch.nn.Module): Second source model
        model3 (torch.nn.Module): Initial merged model
        spec (PermutationSpec): Permutation specification
        perm (Permutation): Permutation mapping
        costs (dict): Cost matrices from permutation matching
        budget_ratios (dict): Ratios for partial merging
        WANDB (bool): Whether to log to Weights & Biases
        MAX_STEPS (int): Maximum number of training steps
        wandb_run: Weights & Biases run object
        separate_classifier (bool, optional): Whether to use separate classifier. Defaults to False.
        merging (str, optional): Merging method. Defaults to 'perm_gradmask'.
        num_classes (int, optional): Number of output classes. Defaults to 1000.
        lr (float, optional): Learning rate. Defaults to 5e-4.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        rn_18 (bool/str, optional): Specific model type flag. Defaults to False.
        
    Returns:
        torch.nn.Module: Trained merged model
    """
    # Get permutation blocks from the specification
    blocks = get_blocks(spec, perm, costs, budget_ratios, False)
    perm_blocks = copy(blocks) 
    for axis, pg in spec.items():
        for ax in pg.state:
            perm_blocks[ax] = perm_blocks[axis]

    
    axes_by_tensor = {}
    for pg in spec.values():
        for ax in pg.state:
            axes_by_tensor.setdefault(ax.key, set()).add(ax.axis)    
    
    # Set up activation capturing hooks for both source models
    activations_model_1 = {}
    activations_model_2 = {}
    m1_hooks = capture_inputs(model1, activations_model_1)
    m2_hooks = capture_inputs(model2, activations_model_2)
    model1.eval()
    model2.eval()
    
    # Extract trainable parameters from merged model
    model3, model3_dict, m3_params = get_model_dict_and_params(model3)
    optimizer = torch.optim.Adam(m3_params, lr=lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEPS)
    
    all_layer_loss = {k: 0 for k in model3_dict.keys()}
    
    # Create gradient masks to enforce structural constraints
    grad_masks = get_gradient_mask(perm_blocks, model3_dict)
    assert len(grad_masks) == len(m3_params)
    
    # Training loop
    tracking_loss = 0.
    wandb_metrics = {}
    for idx, batch in tqdm.tqdm(enumerate(dataloader)):
        if idx > MAX_STEPS: break
        
        # Perform optimization step
        tracking_loss += step(batch, model1, model2, model3_dict, m3_params, optimizer, perm_blocks, all_layer_loss, activations_model_1, activations_model_2, grad_masks, separate_classifier, merging, num_classes, verbose=verbose, model_type=model_type)
        if verbose: print(tracking_loss)
        lr_sched.step()
        
        # Log metrics periodically
        if idx % 20 == 0 and idx:
            print(f"Loss: {tracking_loss / 20:.3f}")
            for k in all_layer_loss.keys():
                wandb_metrics[f"loss_{k}"] = all_layer_loss[k] / 20
                all_layer_loss[k] = 0.

            if WANDB:
                wandb_metrics['step'] = idx
                wandb_metrics['loss'] = tracking_loss / 20
                wandb_run.log(wandb_metrics)

            tracking_loss = 0.
    
    # Update the merged model with trained weights
    m3_sd = model3.state_dict()
    for k,v in model3_dict.items():
        v_sd = v.state_dict()
        for k2 in v_sd.keys():
            m3_sd[f"{k}.{k2}"] = v_sd[k2]
    model3.load_state_dict(m3_sd)
    
    # Clean up hooks
    for hook in m1_hooks:
        hook.remove()
    for hook in m2_hooks:
        hook.remove()
    
    return model3


def get_fc_perm(perm, spec, costs, budget_ratios):
    """
    Get permutation blocks for the fully connected layer.
    
    This is useful for handling the classification head separately
    when evaluating permutation models.
    
    Args:
        perm (Permutation): Permutation mapping
        spec (PermutationSpec): Permutation specification
        costs (dict): Cost matrices from permutation matching
        budget_ratios (dict): Ratios for partial merging
        
    Returns:
        tuple: Permutation blocks for the fully connected layer
    """
    perm_blocks = get_blocks(spec, perm, costs, budget_ratios, False)
    for k,v in spec.items():
        if Axis('fc.weight', 1) in v.state:
            found = True
            break
    if found:
        fc_perm = perm_blocks[k]
    else:
        raise ValueError("No fc perm found")
    return fc_perm


def permute_final_features(features, fc_perm, idx):
    """
    Permute the final layer features according to the specified permutation.
    
    This function is used during evaluation to properly align features 
    from the feature extractor to the appropriate classifier.
    
    Args:
        features (torch.Tensor): Features from the backbone
        fc_perm (tuple): Permutation blocks for the fully connected layer
        idx (int): Index indicating which model's classifier is being used (0 or 1)
        
    Returns:
        torch.Tensor: Permuted features for appropriate classification
    """
    bi1, bi2, bi1c, bi2c = fc_perm
    ni, mi = len(bi1), len(bi1c)
    si12, si1, si2 = (
        slice(0, ni),
        slice(ni, ni + mi),
        slice(ni + mi, ni + 2 * mi),
    )
    
    if idx == 0:
        sliced_feats = torch.cat([features[:,si12], features[:,si1]], dim=1)
        feats = sliced_feats[:,torch.argsort(torch.cat([bi1, bi1c], dim=0))]
    else:
        sliced_feats = torch.cat([features[:,si12], features[:,si2]], dim=1)
        feats = sliced_feats[:,torch.argsort(torch.cat([bi2, bi2c], dim=0))]
    return feats


def eval_perm_model(model, fc, dataloader, num_classes, fc_perm, idx):    
    """
    Evaluate a merged model with a specific classifier.
    
    This function evaluates a merged model with a classifier from one of the
    source models, properly permuting features to align with that classifier.
    
    Args:
        model (torch.nn.Module): Merged model to evaluate
        fc (torch.nn.Module): Classifier from one of the source models
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation
        num_classes (int): Number of output classes
        fc_perm (tuple): Permutation blocks for the fully connected layer
        idx (int): Index indicating which model's classifier is being used (0 or 1)
        
    Returns:
        torch.Tensor: Accuracy of the model on the evaluation dataset
    """
    model.eval()
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).cuda()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            feats = permute_final_features(model(x), fc_perm, idx)
            y_hat = fc(feats)
            acc(y_hat, y)
    return acc.compute()


def train_eval_linear_probe(model, train_dataloader, test_dataloader, num_classes, wandb_run, dataset_name, lr=1e-3, epochs=10):
    """
    Train and evaluate a linear probe on top of a merged model.
    
    This function trains a linear classifier on top of a frozen merged model
    and evaluates its performance. This is useful for assessing transfer learning
    capabilities of the merged model.
    
    Args:
        model (torch.nn.Module): Model to evaluate with linear probing
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training
        test_dataloader (torch.utils.data.DataLoader): DataLoader for evaluation
        num_classes (int): Number of output classes
        wandb_run: Weights & Biases run object for logging
        dataset_name (str): Name of the dataset for logging
        lr (float, optional): Learning rate. Defaults to 1e-3.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        
    Returns:
        torch.nn.Module: Trained linear probe
    """
    model.cuda()
    model.eval()
    out_feats = model(torch.randn(1,3,224,224).cuda()).shape[-1]
    fc = torch.nn.Linear(out_feats, num_classes).cuda()
    optimizer = torch.optim.Adam(fc.parameters(), lr=lr)
    lr_sced = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*len(train_dataloader), eta_min=lr/10)    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.eval()
        fc.train()
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=int(num_classes)).cuda()
        
        total_loss = 0.
        
        for batch in tqdm.tqdm(train_dataloader):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            with torch.no_grad():
                feats = model(x)
            y_hat = fc(feats)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_sced.step()
            train_acc(y_hat, y)
            total_loss += loss.item()
        
        # Log training metrics
        wandb_run.log({f"{dataset_name}_linear_probe_train_acc": train_acc.compute().item(), 
                       f"{dataset_name}_linear_probe_train_loss": loss.item(),
                       'epoch': epoch,
                       f'{dataset_name}_total_loss': total_loss/len(train_dataloader)})
    
    # Evaluation
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=int(num_classes)).cuda()
    for batch in tqdm.tqdm(test_dataloader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            feats = model(x)
        y_hat = fc(feats)
        acc(y_hat, y)
    
    # Log evaluation results
    wandb_run.log({f"{dataset_name}_linear_probe_acc": acc.compute().item()})
    return fc



def eval_whole_model(model, dataloader, num_classes):
    # Simple eval loop
    model.cuda()
    model.eval()
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    for batch in tqdm.tqdm(dataloader):
        x, y = batch
        x = x.cuda()
        y_hat = model(x)
        acc(y_hat.detach().cpu(), y)
    print(acc.compute())
    return acc.compute()