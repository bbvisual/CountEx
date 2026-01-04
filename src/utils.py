import torch
import numpy as np
from transformers import GroundingDinoProcessor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


def prepare_targets(points, caption, shapes, emb_size, device, llmdet_processor):
    gt_points_b = [np.array(points) / np.array(shapes)[::-1]]
    gt_points_b[0] = gt_points_b[0].squeeze(0)

    gt_points = [torch.from_numpy(img_points).float() for img_points in gt_points_b]
    gt_logits = [torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points]

    tokenized = llmdet_processor.tokenizer(caption[0], padding="longest", return_tensors="pt")
    end_idxes = [torch.where(ids == 1012)[0][-1] for ids in tokenized['input_ids']]
    for i, end_idx in enumerate(end_idxes):
        gt_logits[i][:, :end_idx] = 1.0
    caption_sizes = [idx + 2 for idx in end_idxes]

    targets = [{"points": p.to(device), "labels": l.to(device), "caption_size": c}
               for p, l, c in zip(gt_points, gt_logits, caption_sizes)]

    return targets


def post_process_grounded_object_detection(
        outputs,
        box_threshold: float = 0.4,
):
    # for the fine-tuning model, the box threshold should be set to 0.50
    logits, boxes = outputs.logits, outputs.pred_boxes

    probs = torch.sigmoid(logits)  # (batch_size, num_queries, 256)
    scores = torch.max(probs, dim=-1)[0]  # (batch_size, num_queries)

    results = []
    for idx, (s, b, p) in enumerate(zip(scores, boxes, probs)):
        score = s[s > box_threshold]
        box = b[s > box_threshold]
        prob = p[s > box_threshold]
        results.append({"scores": score, "boxes": box})

    return results


class collator:
    def __init__(self, processor=None, use_negative=True):
        model_id = "fushh7/llmdet_swin_tiny_hf"
        self.llmdet_processor = GroundingDinoProcessor.from_pretrained(model_id)
        self.use_negative = use_negative

    def __call__(self, batch):
        # assume batch size is 1
        example = batch[0]
        image = example['image']
        pil_image = example['image']
        w, h = image.size
        pos_caption = example['pos_caption']
        neg_caption = example['neg_caption']
        pos_points = example['pos_points']
        neg_points = example['neg_points']
        pos_count = example['pos_count']
        neg_count = example['neg_count']
        annotated_pos_count = example['annotated_pos_count']
        annotated_neg_count = example['annotated_neg_count']

        if 'type' in example:
            sample_type = example['type']
        else:
            sample_type = 'eval'
        category = example['category']
        image_name = "{}_{}_{}_{}_{}".format(category, pos_caption, neg_caption, pos_count, neg_count)
        pos_llm_det_inputs = self.llmdet_processor(images=image, text=pos_caption, return_tensors="pt", padding=True)
        neg_llm_det_inputs = self.llmdet_processor(images=image, text=neg_caption, return_tensors="pt", padding=True)
        pos_caption = [[pos_caption]]
        neg_caption = [[neg_caption]]
        shapes = [(w, h)]
        pos_points = [pos_points]
        neg_points = [neg_points]

        # exemplars
        if 'positive_exemplars' in example and 'negative_exemplars' in example and example[
            'positive_exemplars'] is not None and example['negative_exemplars'] is not None:
            pos_exemplars = example['positive_exemplars']
            neg_exemplars = example['negative_exemplars']
            img_height, img_width = pil_image.size
            norm_pos_exemplars = []
            norm_neg_exemplars = []
            exemplar_valid = True
            for exemplars in pos_exemplars:
                tly, tlx, bry, brx = exemplars
                tlx = tlx / img_width
                tly = tly / img_height
                brx = brx / img_width
                bry = bry / img_height
                if tlx < 0 or tly < 0 or tlx > 1.0 or tly > 1.0:
                    exemplar_valid = False
                if brx < 0 or bry < 0 or brx > 1.0 or bry > 1.0:
                    exemplar_valid = False
                if tlx >= brx or tly >= bry:
                    exemplar_valid = False
                tlx = max(tlx, 0)
                tly = max(tly, 0)
                tly = min(tly, 1 - 1e-4)
                tlx = min(tlx, 1 - 1e-4)
                brx = min(brx, 1)
                bry = min(bry, 1)
                brx = max(brx, tlx)
                bry = max(bry, tly)
                assert tlx >= 0 and tly >= 0 and brx <= 1 and bry <= 1 and tlx <= brx and tly <= bry, f"tlx: {tlx}, tly: {tly}, brx: {brx}, bry: {bry}"
                norm_pos_exemplars.append([tlx, tly, brx, bry])
            for exemplars in neg_exemplars:
                tly, tlx, bry, brx = exemplars
                tlx = tlx / img_width
                tly = tly / img_height
                brx = brx / img_width
                bry = bry / img_height
                if tlx < 0 or tly < 0 or tlx > 1.0 or tly > 1.0:
                    exemplar_valid = False
                if brx < 0 or bry < 0 or brx > 1.0 or bry > 1.0:
                    exemplar_valid = False
                if tlx >= brx or tly >= bry:
                    exemplar_valid = False
                tlx = max(tlx, 0)
                tly = max(tly, 0)
                tly = min(tly, 1 - 1e-4)
                tlx = min(tlx, 1 - 1e-4)
                brx = min(brx, 1)
                bry = min(bry, 1)
                brx = max(brx, tlx)
                bry = max(bry, tly)
                assert tlx >= 0 and tly >= 0 and brx <= 1 and bry <= 1 and tlx <= brx and tly <= bry, f"tlx: {tlx}, tly: {tly}, brx: {brx}, bry: {bry}"
                norm_neg_exemplars.append([tlx, tly, brx, bry])

            if exemplar_valid:
                pos_exemplars = [torch.from_numpy(np.array(exemplars)).float() for exemplars in norm_pos_exemplars]
                neg_exemplars = [torch.from_numpy(np.array(exemplars)).float() for exemplars in norm_neg_exemplars]
                pos_exemplars = torch.stack(pos_exemplars)
                neg_exemplars = torch.stack(neg_exemplars)
                batch_dict = {
                    'pos_llm_det_inputs': pos_llm_det_inputs,
                    'neg_llm_det_inputs': neg_llm_det_inputs,
                    'pos_caption': pos_caption,
                    'neg_caption': neg_caption,
                    'shapes': shapes,
                    'pos_points': pos_points,
                    'neg_points': neg_points,
                    'pos_count': pos_count,
                    'neg_count': neg_count,
                    'annotated_pos_count': annotated_pos_count,
                    'annotated_neg_count': annotated_neg_count,
                    'image': pil_image,
                    'category': category,
                    'type': sample_type,
                    'pos_exemplars': pos_exemplars,
                    'neg_exemplars': neg_exemplars,
                    'image_name': image_name,
                }
            else:
                batch_dict = {
                    'pos_llm_det_inputs': pos_llm_det_inputs,
                    'neg_llm_det_inputs': neg_llm_det_inputs,
                    'pos_caption': pos_caption,
                    'neg_caption': neg_caption,
                    'shapes': shapes,
                    'pos_points': pos_points,
                    'neg_points': neg_points,
                    'pos_count': pos_count,
                    'neg_count': neg_count,
                    'annotated_pos_count': annotated_pos_count,
                    'annotated_neg_count': annotated_neg_count,
                    'image': pil_image,
                    'category': category,
                    'type': sample_type,
                    'image_name': image_name,
                }
        else:
            batch_dict = {
                'pos_llm_det_inputs': pos_llm_det_inputs,
                'neg_llm_det_inputs': neg_llm_det_inputs,
                'pos_caption': pos_caption,
                'neg_caption': neg_caption,
                'shapes': shapes,
                'pos_points': pos_points,
                'neg_points': neg_points,
                'pos_count': pos_count,
                'neg_count': neg_count,
                'annotated_pos_count': annotated_pos_count,
                'annotated_neg_count': annotated_neg_count,
                'image': pil_image,
                'category': category,
                'type': sample_type,
                'image_name': image_name,
            }

        return batch_dict


import torch.distributed as dist


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def build_dataset(data_args):
    from datasets import load_dataset, concatenate_datasets, load_from_disk

    categories = ["FOO", "FUN", "OFF", "OTR", "HOU"]

    if data_args.data_split not in categories:
        rank0_print(f"Warning: Invalid data_split '{data_args.data_split}'. Switching to 'all' mode.")
        data_args.data_split = "all"

    if data_args.data_split == "all":
        train_dataset = load_dataset(data_args.train_data_path)
        train_dataset = concatenate_datasets(
            [train_dataset["FOO"], train_dataset["FUN"], train_dataset["OFF"], train_dataset["OTR"],
             train_dataset["HOU"]])

        val_dataset = load_dataset(data_args.val_data_path)
        val_dataset = concatenate_datasets(
            [val_dataset["FOO"], val_dataset["FUN"], val_dataset["OFF"], val_dataset["OTR"], val_dataset["HOU"]])

        test_dataset = load_dataset(data_args.test_data_path)
        test_dataset = concatenate_datasets(
            [test_dataset["FOO"], test_dataset["FUN"], test_dataset["OFF"], test_dataset["OTR"], test_dataset["HOU"]])

        weakly_supervised_data = load_dataset(data_args.weakly_supervised_data_path)
        weakly_supervised_data = concatenate_datasets(
            [weakly_supervised_data["FOO"], weakly_supervised_data["FUN"], weakly_supervised_data["OFF"],
             weakly_supervised_data["OTR"], weakly_supervised_data["HOU"]])

        rank0_print("Using 'all' mode: all categories for train/val/test")

    else:
        test_category = data_args.data_split
        train_categories = [cat for cat in categories if cat != test_category]
        train_dataset = load_dataset(data_args.train_data_path)
        print(train_categories, train_dataset.keys())
        train_datasets = [train_dataset[cat] for cat in train_categories]
        train_dataset = concatenate_datasets(train_datasets)

        weakly_supervised_data = load_dataset(data_args.weakly_supervised_data_path)
        weakly_supervised_data = [weakly_supervised_data[cat] for cat in train_categories]
        weakly_supervised_data = concatenate_datasets(weakly_supervised_data)

        val_dataset = load_dataset(data_args.val_data_path)
        val_dataset = val_dataset[test_category]

        test_dataset = load_dataset(data_args.test_data_path)
        test_dataset = test_dataset[test_category]

        rank0_print(f"Cross-validation mode: using {train_categories} for train, {test_category} for val/test")

    rank0_print('train_dataset: ', len(train_dataset))
    rank0_print('val_dataset: ', len(val_dataset))
    rank0_print('test_dataset: ', len(test_dataset))
    rank0_print('weakly_supervised_data: ', len(weakly_supervised_data))

    return train_dataset, val_dataset, test_dataset, weakly_supervised_data


def generate_pseudo_density_map(points_norm: torch.Tensor,
                                output_size: tuple[int, int],
                                sigma: float = 4.0,
                                normalize: bool = True) -> torch.Tensor:
    device = points_norm.device
    H, W = output_size
    N = points_norm.shape[0]

    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

    pts_px = points_norm.clone()
    pts_px[:, 0] *= (W - 1)  # x
    pts_px[:, 1] *= (H - 1)  # y

    dx = grid_x.unsqueeze(0) - pts_px[:, 0].view(-1, 1, 1)  # (N, H, W)
    dy = grid_y.unsqueeze(0) - pts_px[:, 1].view(-1, 1, 1)  # (N, H, W)
    dist2 = dx ** 2 + dy ** 2
    gaussians = torch.exp(-dist2 / (2 * sigma ** 2))  # (N, H, W)
    density_map = gaussians.sum(dim=0, keepdim=True)  # (1, H, W)

    if normalize and N > 0:
        density_map = density_map * (N / density_map.sum())

    return density_map.unsqueeze(0)


def show_density_map(density_map: torch.Tensor,
                     points_norm: torch.Tensor | None = None,
                     figsize: tuple[int, int] = (6, 8),
                     cmap: str = "jet") -> None:
    dm = density_map.squeeze().detach().cpu().numpy()  # (H, W)
    H, W = dm.shape

    plt.figure(figsize=figsize)
    plt.imshow(dm, cmap=cmap, origin="upper")
    plt.colorbar(label="Density")

    if points_norm is not None and points_norm.numel() > 0:
        pts = points_norm.detach().cpu().numpy()
        xs = pts[:, 0] * (W - 1)
        ys = pts[:, 1] * (H - 1)
        plt.scatter(xs, ys, c="white", s=12, edgecolors="black", linewidths=0.5)

    plt.title(f"Density map  (sum = {dm.sum():.2f})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_image_with_density(pil_img: Image.Image,
                            density_map: torch.Tensor,
                            points_norm: torch.Tensor | None = None,
                            cmap: str = "jet",
                            alpha: float = 0.45,
                            figsize: tuple[int, int] = (6, 8)) -> None:
    dm = density_map.squeeze().detach().cpu().numpy()  # (H, W)
    H, W = dm.shape

    img_resized = pil_img.resize((W, H), Image.BILINEAR)  # or LANCZOS
    img_np = np.asarray(img_resized)

    plt.figure(figsize=figsize)
    plt.imshow(img_np, origin="upper")
    plt.imshow(dm, cmap=cmap, alpha=alpha, origin="upper")

    if points_norm is not None and points_norm.numel() > 0:
        pts = points_norm.detach().cpu().numpy()
        xs = pts[:, 0] * (W - 1)
        ys = pts[:, 1] * (H - 1)
        plt.scatter(xs, ys, c="white", s=12, edgecolors="black", linewidths=0.5)

    plt.title(f"Overlay (density sum = {dm.sum():.2f})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def build_point_count_map(feat_maps: torch.Tensor,
                          pts_norm_list: list[torch.Tensor]) -> torch.Tensor:
    assert feat_maps.dim() == 4, "expect NHWC: (B,H,W,D)"
    B, H, W, _ = feat_maps.shape
    device = feat_maps.device

    count_map = torch.zeros((B, H, W), dtype=torch.float32, device=device)

    for b in range(B):
        pts = pts_norm_list[b].to(device).float()  # (Ni, 2)
        if pts.numel() == 0:
            continue

        idx_xy = (pts * torch.tensor([W, H], device=device)).long()
        idx_xy[..., 0].clamp_(0, W - 1)  # x
        idx_xy[..., 1].clamp_(0, H - 1)  # y

        lin_idx = idx_xy[:, 1] * W + idx_xy[:, 0]  # (Ni,)
        one = torch.ones_like(lin_idx, dtype=torch.float32)

        flat = torch.zeros(H * W, dtype=torch.float32, device=device)
        flat.scatter_add_(0, lin_idx, one)
        count_map[b] = flat.view(H, W)

    return count_map


import torch
import torch.nn.functional as F


def extract_pos_tokens_single(feat_maps: torch.Tensor,
                              count_map: torch.Tensor):
    B, H, W, D = feat_maps.shape
    feat = feat_maps[0]  # (H,W,D)
    cnt = count_map[0]  # (H,W)
    pos_mask = cnt > 0  # Bool (H,W)
    if pos_mask.sum() == 0:
        empty = torch.empty(0, device=feat.device)
        return empty.reshape(0, D), empty.long()
    pos_tokens = feat[pos_mask]  # (N_pos, D)
    y_idx, x_idx = torch.nonzero(pos_mask, as_tuple=True)
    lin_index = y_idx * W + x_idx  # (N_pos,)
    return pos_tokens, lin_index


def filter_overlap(pos_tok, lin_pos, neg_tok, lin_neg):
    pos_only_mask = ~torch.isin(lin_pos, lin_neg)
    neg_only_mask = ~torch.isin(lin_neg, lin_pos)
    return pos_tok[pos_only_mask], neg_tok[neg_only_mask]


# ------------------------------------------------------------
# 2) supervised contrastive loss
# ------------------------------------------------------------
def supcon_pos_neg(pos_tokens, neg_tokens, temperature=0.07):
    """
    pos_tokens : (Np, D)  Pos token
    neg_tokens : (Nn, D)  Neg token
    """
    if pos_tokens.numel() == 0 or neg_tokens.numel() == 0:
        return torch.tensor(0., device=pos_tokens.device, requires_grad=True)
    pos_tokens = F.normalize(pos_tokens, dim=-1)
    neg_tokens = F.normalize(neg_tokens, dim=-1)
    feats = torch.cat([pos_tokens, neg_tokens], dim=0)  # (N, D)
    labels = torch.cat([torch.zeros(len(pos_tokens), device=feats.device, dtype=torch.long),
                        torch.ones(len(neg_tokens), device=feats.device, dtype=torch.long)], dim=0)  # (N,)
    logits = feats @ feats.T / temperature  # (N, N)
    logits.fill_diagonal_(-1e4)
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
    mask_pos.fill_diagonal_(False)
    exp_logits = logits.exp()
    denom = exp_logits.sum(dim=1, keepdim=True)  # Σ_{a≠i} exp
    log_prob = logits - denom.log()  # log softmax
    loss_i = -(mask_pos * log_prob).sum(1) / mask_pos.sum(1).clamp_min(1)
    loss = loss_i.mean()
    return loss


def build_point_count_map(feat_maps: torch.Tensor,
                          pts_norm_list: list[torch.Tensor]) -> torch.Tensor:
    assert feat_maps.dim() == 4, "expect NHWC: (B,H,W,D)"
    B, H, W, _ = feat_maps.shape
    device = feat_maps.device

    count_map = torch.zeros((B, H, W), dtype=torch.float32, device=device)

    for b in range(B):
        pts = pts_norm_list[b].to(device).float()  # (Ni, 2)
        if pts.numel() == 0:
            continue

        idx_xy = (pts * torch.tensor([W, H], device=device)).long()
        idx_xy[..., 0].clamp_(0, W - 1)  # x
        idx_xy[..., 1].clamp_(0, H - 1)  # y

        lin_idx = idx_xy[:, 1] * W + idx_xy[:, 0]  # (Ni,)
        one = torch.ones_like(lin_idx, dtype=torch.float32)

        flat = torch.zeros(H * W, dtype=torch.float32, device=device)
        flat.scatter_add_(0, lin_idx, one)
        count_map[b] = flat.view(H, W)

    return count_map


def interval_huber(pred_cnt: torch.Tensor,
                   gt_cnt:   torch.Tensor,
                   rho: float = 0.1,
                   delta_min: float = 1.0) -> torch.Tensor:
    """
    ε-insensitive + Huber:
        δ = max(rho * gt, delta_min)
        err <= δ          : 0                  
        δ < err <= 2δ     : 0.5*(err-δ)^2/δ 
        err > 2δ          : err - 1.5*δ 
    pred_cnt, gt_cnt 形状均为 (B,)
    """
    pred_cnt = pred_cnt.unsqueeze(0)
    gt_cnt = gt_cnt.unsqueeze(0)
    # print(f"pred_cnt: {pred_cnt.shape}, gt_cnt: {gt_cnt.shape}")
    delta = torch.clamp(rho * gt_cnt, min=delta_min)
    err   = torch.abs(pred_cnt - gt_cnt)

    loss  = torch.zeros_like(err)
    mask1 = (err > delta) & (err <= 2 * delta)
    mask2 = (err > 2 * delta)

    loss[mask1] = 0.5 * (err[mask1] - delta[mask1]) ** 2 / delta[mask1]
    loss[mask2] = err[mask2] - 1.5 * delta[mask2]

    return loss.mean()


def relative_huber(pred, target, rho=0.1, eps=1e-6):
    e_rel = torch.abs(pred - target) / (target + eps)
    delta = rho
    loss  = torch.zeros_like(e_rel)

    mask1 = (e_rel > delta) & (e_rel <= 2*delta)
    mask2 = e_rel > 2*delta

    loss[mask1] = 0.5 * (e_rel[mask1] - delta) ** 2 / delta
    loss[mask2] = e_rel[mask2] - 1.5 * delta
    return loss.mean()