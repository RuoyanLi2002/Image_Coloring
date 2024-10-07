import math
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch
import pyjuice as juice
from functools import partial


B = 4

def forward(pc, ns, grey_prob, args, **kwargs):
    grey_prob = torch.from_numpy(grey_prob)
    grey_prob = grey_prob.to(args.device)

    # Set propagation algorithm
    propagation_alg = pc.default_propagation_alg
        
    ## Initialize buffers for forward pass ##
    pc._init_buffer(name = "node_mars", shape = (pc.num_nodes, B), set_value = 0.0)
    pc._init_buffer(name = "element_mars", shape = (pc.num_elements, B), set_value = -torch.inf)
    
    ## Run forward pass ##
    with torch.no_grad():
        # Inner layers
        def _run_inner_layers():
            for layer_id, layer_group in enumerate(pc.inner_layer_groups):
                if layer_id == 0:
                    
                    pc.element_mars = grey_prob.expand_as(pc.element_mars)
                    pc.element_mars = pc.element_mars.contiguous()

                    backward_node_mars = pc.element_mars.clone()
                    
                else:
                    if layer_group.is_prod():
                        # Prod layer
                        layer_group(pc.node_mars, pc.element_mars)

                    elif layer_group.is_sum():
                        # Sum layer
                        layer_group(pc.node_mars, pc.element_mars, pc.params, 
                                    force_use_fp16 = False,
                                    force_use_fp32 = False, 
                                    propagation_alg = propagation_alg, 
                                    **kwargs)


                    else:
                        raise ValueError(f"Unknown layer type {type(layer)}.")

            return backward_node_mars

        backward_node_mars = _run_inner_layers()
        
        lls = pc.node_mars[pc._root_node_range[0]:pc._root_node_range[1],:]
        lls = lls.permute(1, 0)

    ## Add gradient hook for backward pass ##
    def _pc_model_backward_hook(grad, pc, inputs, record_cudagraph, apply_cudagraph, propagation_alg, **kwargs):
        grad = grad.permute(1, 0)
        pc.backward(
            inputs = inputs,
            ll_weights = grad / grad.sum() * grad.size(1),
            compute_param_flows = pc._optim_hyperparams["compute_param_flows"], 
            flows_memory = pc._optim_hyperparams["flows_memory"],
            record_cudagraph = record_cudagraph,
            apply_cudagraph = apply_cudagraph,
            propagation_alg = propagation_alg,
            **kwargs
        )

        return None

    if torch.is_grad_enabled():
        lls.requires_grad = True
        lls.register_hook(
            partial(
                _pc_model_backward_hook, 
                pc = pc, 
                inputs = grey_prob, 
                record_cudagraph = False, 
                apply_cudagraph = True,
                propagation_alg = propagation_alg,
                **kwargs
            )
        )

    return lls.clone()[0, :], backward_node_mars


def backward(pc, backward_node_mars, ll_weights = None, logspace_flows = False, allow_modify_flows = False, compute_param_flows = True, **kwargs):
    assert pc.node_mars is not None and pc.element_mars is not None, "Should run forward path first."

    ## Initialize buffers for backward pass ##
    pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, B), set_value = 0.0 if not logspace_flows else -float("inf"))
    pc._init_buffer(name = "element_flows", shape = (pc.num_elements, B), set_value = 0.0 if not logspace_flows else -float("inf"))


    # Set root node flows
    def _set_root_node_flows():
        nonlocal ll_weights
        nonlocal logspace_flows
        if ll_weights is None:
            root_flows = 1.0 if not logspace_flows else 0.0
            pc.node_flows[pc._root_node_range[0]:pc._root_node_range[1],:] = root_flows
        else:
            if ll_weights.dim() == 1:
                ll_weights = ll_weights.unsqueeze(1)

            assert ll_weights.size(0) == pc.num_root_nodes

            root_flows = ll_weights if not logspace_flows else ll_weights.log()
            pc.node_flows[pc._root_node_range[0]:pc._root_node_range[1],:] = root_flows

    _set_root_node_flows()
    
    ## Initialize parameter flows ##
    if compute_param_flows:
        pc.init_param_flows(flows_memory = 1.0)

    ## Run backward pass ##
    with torch.no_grad():
        # Inner layers
        def _run_inner_layers(backward_node_mars):
            # Backward pass for inner layers
            for layer_id in range(len(pc.inner_layer_groups) - 1, -1, -1):
                layer_group = pc.inner_layer_groups[layer_id]

                if layer_id == 0:
                    break

                if layer_group.is_prod():
                    # Prod layer
                    layer_group.backward(pc.node_flows, pc.element_flows, logspace_flows = logspace_flows)

                elif layer_group.is_sum():
                    # Sum layer

                    # First recompute the previous product layer
                    if layer_id == 1:
                        pc.element_mars = backward_node_mars.clone()
                        pc.element_mars = pc.element_mars.contiguous()
                    else:
                        pc.inner_layer_groups[layer_id-1].forward(pc.node_mars, pc.element_mars, _for_backward = True)
                   

                    # Backward sum layer
                    propagation_alg = "LL"
                    layer_group.backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params, 
                                        param_flows = pc.param_flows,
                                        allow_modify_flows = allow_modify_flows, 
                                        propagation_alg = propagation_alg, 
                                        logspace_flows = logspace_flows, negate_pflows = allow_modify_flows, **kwargs)

                else:
                    raise ValueError(f"Unknown layer type {type(layer)}.")

                    
                # print(f"self.node_flows: {pc.node_flows}")
                # print(f"self.element_flows: {pc.element_flows}")

            
        _run_inner_layers(backward_node_mars)

    return pc.element_flows[:, 0]