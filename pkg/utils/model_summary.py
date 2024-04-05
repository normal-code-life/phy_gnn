"""This is a copy from: pytorch-model-summary."""

from collections import OrderedDict

import torch
import torch.nn as nn


def summary(
    model,
    *inputs,
    batch_size=-1,
    show_input=False,
    show_hierarchical=False,
    print_summary=False,
    max_depth=1,
    show_parent_layers=False,
):
    max_depth = max_depth if max_depth is not None else 9999

    def build_module_tree(module):
        def _in(module, id_parent, depth):
            for key, c in module.named_children():
                # ModuleList and Sequential do not count as valid layers
                if isinstance(c, (nn.ModuleList, nn.Sequential)):
                    _in(c, id_parent, depth)
                else:
                    _module_name = str(key).split(".")[-1].split("'")[0]
                    _parent_layers = (
                        f'{module_summary[id_parent].get("parent_layers")}'
                        f'{"/" if module_summary[id_parent].get("parent_layers") != "" else ""}'
                        f'{module_summary[id_parent]["module_name"]}'
                    )

                    module_summary[id(c)] = {
                        "module_name": _module_name,
                        "parent_layers": _parent_layers,
                        "id_parent": id_parent,
                        "depth": depth,
                        "n_children": 0,
                    }

                    module_summary[id_parent]["n_children"] += 1

                    _in(c, id(c), depth + 1)

        # Defining summary for the main module
        module_summary[id(module)] = {
            "module_name": str(module.__class__).split(".")[-1].split("'")[0],
            "parent_layers": "",
            "id_parent": None,
            "depth": 0,
            "n_children": 0,
        }

        _in(module, id_parent=id(module), depth=1)

        # Defining layers that will be printed
        for k, v in module_summary.items():
            module_summary[k]["show"] = v["depth"] == max_depth or (v["depth"] < max_depth and v["n_children"] == 0)

    def register_hook(module):
        def shapes(x):
            _lst = list()

            def _shapes(_):
                if isinstance(_, torch.Tensor):
                    _lst.append(list(_.size()))
                elif isinstance(_, (tuple, list)):
                    for _x in _:
                        _shapes(_x)
                else:
                    # TODO: decide what to do when there is an input which is not a tensor
                    raise Exception("Object not supported")

            _shapes(x)

            return _lst

        def hook(module, input, output=None):
            if id(module) in module_mapped:
                return

            module_mapped.add(id(module))
            module_name = module_summary.get(id(module)).get("module_name")
            module_idx = len(summary)

            m_key = "%s-%i" % (module_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["parent_layers"] = module_summary.get(id(module)).get("parent_layers")

            summary[m_key]["input_shape"] = shapes(input) if len(input) != 0 else input

            if show_input is False and output is not None:
                summary[m_key]["output_shape"] = shapes(output)

            params = 0
            params_trainable = 0
            trainable = False
            for m in module.parameters():
                _params = torch.prod(torch.LongTensor(list(m.size())))
                params += _params
                params_trainable += _params if m.requires_grad else 0
                # if any parameter is trainable, then module is trainable
                trainable = trainable or m.requires_grad

            summary[m_key]["nb_params"] = params
            summary[m_key]["nb_params_trainable"] = params_trainable
            summary[m_key]["trainable"] = trainable

        _map_module = module_summary.get(id(module), None)
        if _map_module is not None and _map_module.get("show"):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    module_summary = dict()
    module_mapped = set()
    hooks = []

    # register id of parent modules
    build_module_tree(model)

    # register hook
    model.apply(register_hook)

    model_training = model.training

    model.eval()
    model(*inputs)

    if model_training:
        model.train()

    # remove these hooks
    for h in hooks:
        h.remove()

    # params to format output - dynamic width
    _key_shape = "input_shape" if show_input else "output_shape"
    _len_str_parent = max([len(v["parent_layers"]) for v in summary.values()] + [13]) + 3
    _len_str_layer = max([len(layer) for layer in summary.keys()] + [15]) + 3
    _len_str_shapes = (
        max([len(", ".join([str(_) for _ in summary[layer][_key_shape]])) for layer in summary] + [15]) + 3
    )
    _len_line = 35 + _len_str_parent * int(show_parent_layers) + _len_str_layer + _len_str_shapes
    fmt = ("{:>%d} " % _len_str_parent if show_parent_layers else "") + "{:>%d}  {:>%d} {:>15} {:>15}" % (
        _len_str_layer,
        _len_str_shapes,
    )

    """ starting to build output text """

    # Table header
    lines = list()
    lines.append("-" * _len_line)
    _fmt_args = ("Parent Layers",) if show_parent_layers else ()
    _fmt_args += ("Layer (type)", f'{"Input" if show_input else "Output"} Shape', "Param #", "Tr. Param #")
    lines.append(fmt.format(*_fmt_args))
    lines.append("=" * _len_line)

    total_params = 0
    trainable_params = 0
    for layer in summary:
        # Table content (for each layer)
        _fmt_args = (summary[layer]["parent_layers"],) if show_parent_layers else ()
        _fmt_args += (
            layer,
            ", ".join([str(_) for _ in summary[layer][_key_shape]]),
            "{0:,}".format(summary[layer]["nb_params"]),
            "{0:,}".format(summary[layer]["nb_params_trainable"]),
        )
        line_new = fmt.format(*_fmt_args)
        lines.append(line_new)

        total_params += summary[layer]["nb_params"]
        trainable_params += summary[layer]["nb_params_trainable"]

    # Table footer
    lines.append("=" * _len_line)
    lines.append("Total params: {0:,}".format(total_params))
    lines.append("Trainable params: {0:,}".format(trainable_params))
    lines.append("Non-trainable params: {0:,}".format(total_params - trainable_params))
    if batch_size != -1:
        lines.append("Batch size: {0:,}".format(batch_size))
    lines.append("-" * _len_line)

    # if show_hierarchical:
    #     h_summary, _ = hierarchical_summary(model, print_summary=False)
    #     lines.append('\n')
    #     lines.append(h_summary)

    str_summary = "\n".join(lines)
    if print_summary:
        print(str_summary)

    return str_summary
