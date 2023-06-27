import itertools
import os.path

import numpy as np
from data_utils import take, EXIST, COR, DataGenerator, ims, take_dict, SIZE
from data_utils.image import draw_images, add_text, draw_images2, draw_images4, get_image_color, add_frame
from funcy import identity

from experiment_utils import METRIC
from ml_utils import none, filter_keys, lu, lw, pj, il
from models_utils import is_model
from models_utils import ops as K

from raven_utils.constant import PROPERTY, TARGET, INPUTS, INDEX
from raven_utils.decode import decode_target, target_mask
from raven_utils.render.rendering import render_panels
from raven_utils.render_ import TYPES, SIZES
from raven_utils.uitls import get_val_index

ADD_BOARDS = "add_boards"

LAYOUT = "layout"


def draw_board(
        images,
        target=None,
        predict=None,
        image=None,
        desc=None,
        layout=2,
        mark_query_panel=True,
):
    # if image != "target" and predict is not None:
    #     image = images[predict + 8:predict + 9]
    # elif images is None and target is not None:
    #     image = images[target:target + 1]
    # image = False to not draw anything
    if image is None:
        if predict is not None:
            image = images[predict + 8:predict + 9]
        else:
            image = images[target:target + 1]
    border = [
                 {
                     COR: target - 8,
                     EXIST: list(range(4)) if predict is None else (1, 3),
                     SIZE: 6,
                 }
             ] + [
        {
            COR: p,
            EXIST: (0, 2),
            SIZE: 6,
        } for p in none(lw(predict))
             ]

    boards = []
    boards.append(
        draw_images2(
            np.concatenate(
                [
                    images[:8],
                    image[None] if len(image.shape) != 3 else image
                ]
            )
            if image is not None
            else images[:8],
            space=True,
            border=8 if mark_query_panel else None,
            # border_kwargs={SIZE: 1},
            default_border=1,
        )
    )
    # ims(boards[-1][None, :, :, 0])
    layout_kwargs = {
        LAYOUT: 2,
        "grid": False,
        "break": 40,
        "break_color": "max",
        "fill_rest": False
    }
    if isinstance(layout, dict):
        layout_kwargs = {**layout_kwargs, **layout}
        layout = layout.get(LAYOUT, None)
        add_boards = layout_kwargs.get(ADD_BOARDS, None)
        break_ = layout_kwargs.get("break", None)
        break_color = layout_kwargs.get("break_color", None)
    else:
        add_boards = None
        break_ = 40
        break_color = "max"
        # break_color = "min"
    if layout > 0:
        if layout == 2:
            i = draw_images2(
                images[8:],
                col=4,
                border=border,
                border_kwargs={SIZE: 1},
                default_border=True,
            )
        else:
            i = draw_images2(images[8:], border=border, default_border=True)
        if break_:
            # for now break for standard only below and for add_boards between
            # if add_boards:
            #     i = np.concatenate([np.zeros([i.shape[0],break_, 1]), i], axis=1)
            # else:
            i = add_frame(
                image=i,
                direction='t',
                size=break_,
                color=break_color,
            )
        boards.append(i)
        # ims(i[None, :, :, 0])
        # ims(full_board[None, :, :, 0])
        # ims(i)

    else:
        boards.append(
            draw_images(np.concatenate([images[8:], predict]) if predict is not None else images[8:], column=4,
                        border=target - 8))
    # the break is not consistent for
    if break_:
        add_boards = [
            add_frame(
                image=i,
                direction='t',
                size=break_,
                color=break_color,
            )
            for i in lw(add_boards)
        ]
        # add_boards = [
        #     np.concatenate([
        #         np.zeros([
        #             i.shape[0],
        #             break_,
        #             1
        #         ]),
        #         i
        #     ],
        #         axis=1
        #     )
        #     for i in lw(add_boards)
        # ]
        # ims(full_board[None, :, :, 0])
        # ims(boards[0][None, :, :, 0])
    boards.extend(lw(add_boards))
    full_board = draw_images2(boards, **layout_kwargs)
    if desc:
        # not working
        # from data_utils.draw import add_text
        # ims(np.array(add_text(desc, full_board)))
        full_board = add_text(full_board, desc)
    return full_board


def draw_boards(images, target=None, predict=None, image=None, desc=None, layout=2):
    boards = []
    for i, im in enumerate(images):
        # boards.append(draw_board(im, target[i][0] if target is not None else None,
        #                          predict[i] if predict is not None else None,
        #                          image[i] if image is not None else None,
        #                          desc[i] if desc is not None else None, layout=layout))
        boards.append(
            np.asarray(
                draw_board(
                    im, target[i][0] if target is not None else None,
                    predict[i] if predict is not None else None,
                    image[i] if image is not None else None,
                    desc[i] if desc is not None else None,
                    layout=layout[i] if il(layout) else layout
                ),
                dtype=np.uint8
            )
        )
    return boards


def draw_from_generator(generator, predict=None, no=1, indexes=None, layout=1):
    data, _ = val_sample(generator, no, indexes)
    return draw_raven(data, predict=predict, pre_fn=generator.data.data["inputs"].fn, layout=layout)


def val_sample(generator, no=1, indexes=None):
    if indexes is None:
        indexes = get_val_index(base=no)
    data = generator.data[indexes]
    return data, indexes


def render_from_model(data, predict, pre_fn=identity):
    data = filter_keys(data, PROPERTY, reverse=True)
    if is_model(predict):
        predict = predict(data)
    pro = np.array(target_mask(predict['predict_mask'].numpy()) * predict["predict"].numpy(), dtype=np.int8)
    return pre_fn(render_panels(pro, target=False)[None])[0]


def show_raven(data, *args, show=True, path=None, max_=2, **kwargs):
    if isinstance(data, DataGenerator):
        result = draw_from_generator(data, *args, **kwargs)
    else:
        if max_ is not None:
            data = take_dict(data, slice(max_))
        result = draw_raven(data, *args, **kwargs)
    result = draw_images2(result, row=1)
    if show:
        ims(result, path=path)
    return result


# desc not working
def draw_raven(
        data,
        predict=None,
        show=False,
        pre_fn=identity,
        layout=2,
        metrics=None,
        pro=None,
        predict_index=None,
        angle=0
):
    # metrics will be overwritten by model in predict if model output metrics
    if is_model(predict):
        d = filter_keys(data, PROPERTY, reverse=True)
        # tmp change
        res = predict(d)
        pro = np.array(target_mask(res['predict_mask'].numpy()) * res["predict"].numpy(), dtype=np.int8)
        if METRIC in res:
            metrics = res[METRIC]
            for m in ["prob_metric", "kl_metric", "kl_2_metric"]:
                if m in res:
                    metrics[m] = lu(res[m])
            metrics = {k: v.numpy() for k, v in metrics.items() if hasattr(v, "shape") and len(v.shape) == 1}
        predict = pre_fn(render_panels(pro, target=False, angle=angle)[None])[0]
        # render_panels(pro, target=False)
        # from data_utils import ims
        # ims(1 - predict[0])
    # if target is not None:
    target_index = data[INDEX]
    target = K.gather(data[TARGET], target_index[:, 0])
    images = data[INPUTS]
    # np.equal(res['predict'], pro[:,:102]).sum()

    if hasattr(predict, "shape"):
        if len(predict.shape) > 2:
            # images
            image = predict
            # todo create index and output based on image
            predict = pro
            # predict_index = None
        elif len(predict.shape) == 2:
            image = render_panels(predict, target=False, angle=angle)
            # Create index based on predict.
            # predict_index = None
        else:
            image = K.gather(images, predict + 8)
            predict_index = predict
            predict = K.gather(data[TARGET], predict + 8)
    else:
        image = K.gather(images, target_index[:, 0])
        predict_index = None
        predict = None

    # elif not(hasattr(target,"shape") and len(target.shape) > 3):
    #     if hasattr(target,"shape") and target.shape[-1] == OUTPUT_SIZE:
    #         pro = target
    #         predict = render_panels(pro)
    #     elif hasattr(target,"shape") and target.shape[-1] == FEATURE_NO:
    #         # pro = target
    #         pro = np.zeros([no, OUTPUT_SIZE], dtype="int")
    #     else:
    #         pro = np.zeros([no, OUTPUT_SIZE], dtype="int")
    #         # predict = [None] * no
    #         predict = render_panels(data[TARGET])

    image = draw_boards(images, target=target_index, predict=predict_index, image=image, desc=None,
                        layout=layout)
    if PROPERTY not in data:
        if show:
            if il(show):
                show = lambda x, y=show: os.path.expanduser(y[x])
            elif not isinstance(show, str):
                show = lambda x: None
            else:
                show = lambda x, y=show: pj(y, f"rav_{x}")
            [ims(im[None, ..., 0], path=show(i)) for i, im in enumerate(image)]
        return image
    all_rules = extract_rules(data[PROPERTY])
    target_desc = get_desc(target)
    if predict is not None:
        predict_desc = get_desc(predict)
    else:
        predict_desc = [None] * len(target_desc)
    for i, (a, po, to) in enumerate(zip(all_rules, predict_desc, target_desc)):
        # fl(predict_desc[-1])
        if po is None:
            po = [None] * len(to)
        for p, t in list(itertools.zip_longest(po, to, fillvalue="")):
            a.extend(
                [" ".join([str(i) for i in t])] + (
                    [" ".join([str(i) for i in p]), ""] if p is not None else []
                )
            )
        if metrics:
            a.extend([""] + [f"{k}: {v[i]}" for k, v in metrics.items()])
        # a.extend([""] + [] + [""] + [" ".join(fl(p))])

    # image = draw_boards(data[INPUTS], target=data["index"], predict=predict[:no], desc=all_rules, no=no, layer=layer)

    # image = draw_boards(images, target=target_index, predict=predict_index, image=image, desc=all_rules,
    #                     layout=layout)
    result = [(i, j) for i, j in zip(image, all_rules)]
    if show:
        if il(show):
            show = lambda x, y=show: os.path.expanduser(y[x])
        elif not isinstance(show, str):
            show = lambda x: None
        else:
            show = lambda x, y=show: pj(y, f"rav_{x}")
        [ims(im[None, ..., 0], description=j, path=show(i)) for i, (im, j) in enumerate(result)]
    # id lu needed?
    # return lu(result)
    return result


def extract_rules(data):
    all_rules = []
    for d in data:
        rules = []
        for j, rule_group in enumerate(d.findAll("Rule_Group")):
            # rules_all.append(rule_group['id'])
            for j, rule in enumerate(rule_group.findAll("Rule")):
                rules.append(f"{rule['attr']} - {rule['name']}")
            rules.append("")
        all_rules.append(rules)
    return all_rules


def get_desc(target, exist=None, types=TYPES, sizes=SIZES):
    decoded = decode_target(target)
    exist = decoded[1] if exist is None else exist
    taken = np.stack(take(decoded[2], np.array(exist, dtype=bool))).T

    figures_no = np.sum(exist, axis=-1)
    desc = np.split(taken, np.cumsum(figures_no))[:-1]
    # figures_no = np.sum(exist, axis=-1)
    # div = np.split(desc, np.cumsum(figures_no))[:-1]
    result = []
    for pd in desc:
        r = []
        for p in pd:
            r.append([p[0], sizes[p[1]], types[p[2]]])
        result.append(r)

    return result
