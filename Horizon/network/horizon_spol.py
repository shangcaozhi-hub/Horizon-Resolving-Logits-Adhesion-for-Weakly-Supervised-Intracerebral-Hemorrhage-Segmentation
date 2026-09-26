import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from einops import rearrange
from network.spol import SPOL_pretrain, PrototypeSupportBank, dilated_skull_mask


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def _checkpoint_config(checkpoint, num_cls, backbone_config):
    """Rebuild every bank before selecting one for downstream similarity."""
    state = checkpoint["models"]
    expected_projections = {name: (out_channels, in_channels)
                            for name, (in_channels, out_channels)
                            in SPOL_pretrain.PROTOTYPE_CHANNELS.items()}
    if any(tuple(state.get(f"scale_projections.{name}.weight", torch.empty(0)).shape)
           != (*shape, 1, 1) for name, shape in expected_projections.items()):
        raise ValueError(f"Checkpoint projections do not match the current SPOL topology: {expected_projections}")
    saved_args = checkpoint.get("args")
    if saved_args is None and backbone_config is None:
        raise ValueError("Checkpoint lacks training args; supply the complete backbone_config")
    args = (saved_args if isinstance(saved_args, dict) else
            vars(saved_args) if saved_args is not None else {})
    # Infer saved foreground topology, including legacy single-component banks.
    component_pairs = [(checkpoint["models"][key[:-1] + "0"].shape[0], value.shape[0])
                       for key, value in checkpoint["models"].items()
                       if key.endswith(".gmm_means_1")]
    foreground_counts = {fore for _, fore in component_pairs}
    matched_components = all(back == fore for back, fore in component_pairs)
    covariance_ranks = {value.ndim for key, value in checkpoint["models"].items()
                        if ".gmm_covariances_" in key}
    if len(covariance_ranks) > 1 or not covariance_ranks.issubset({2, 3}):
        raise ValueError("Inconsistent GMM covariance shapes in checkpoint")
    if not matched_components and len(foreground_counts) > 1:
        raise ValueError("Checkpoint contains inconsistent foreground component counts")
    # Reconstruct the full training topology, independently of bank selection.
    config = dict(
        num_cls=num_cls,
        num_prototype=args.get("spol_num_prototypes", 20),
        num_components=args.get("spol_components", [1, 2, 3, 4, 5, 10]),
        bank_sizes=args.get("spol_bank_sizes", [500, 1000, 2000]),
        thresholds=args.get("spol_thresholds", [0.8]),
        component_bank_size=args.get("spol_component_bank_size", 1000),
        capacity_components=args.get("spol_capacity_components", 5),
        cluster_bank_size=args.get("spol_cluster_bank_size", 1000),
        train_cluster=not args.get("spol_no_cluster", False),
        cluster_num_prototypes=args.get("spol_cluster_prototypes", 50),
        gmm_init_iterations=args.get("spol_gmm_init_iterations", args.get("spol_gmm_iterations", 50)),
        gmm_covariance_type="full" if covariance_ranks == {3} else "diag",
        gmm_momentum_gamma=args.get("spol_gmm_gamma", 0.9),
        gmm_covariance_shrinkage=args.get("spol_gmm_covariance_shrinkage", 0.25),
        foreground_components=None if matched_components else next(iter(foreground_counts)),
    )
    if backbone_config is not None:
        overrides = dict(backbone_config)
        # Accept legacy configuration aliases for checkpoint reconstruction.
        if "gamma" in overrides:
            overrides.setdefault("gmm_momentum_gamma", overrides.pop("gamma"))
        overrides.pop("gmm_online_statistics", None)
        if "gmm_iterations" in overrides:
            overrides.setdefault("gmm_init_iterations", overrides.pop("gmm_iterations"))
        config.update(overrides)
    return config


class Plug_play(nn.Module):
    """Load complete SPOL weights and all banks before choosing one bank.

    GMM takes each class's maximum similarity over sampled prototypes and component means.
    OT uses saved clustering centers. Evaluation never updates either bank.
    """

    def __init__(self, num_cls=1, num_prototype=None, threshold=0.8,
                 *, checkpoint_path, bank_group="components", bank_size=1000,
                 num_components=3, backbone_config=None, update_bank=False):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "models" not in checkpoint:
            raise ValueError("Expected a SPOL checkpoint containing 'models' and training 'args'")
        config = _checkpoint_config(checkpoint, num_cls, backbone_config)
        if num_prototype is not None:
            config["num_prototype"] = num_prototype
        config["pretrained"] = False
        self.backbone = SPOL_pretrain(**config)
        # Current checkpoints load exactly; legacy inference may lack online state.
        loaded = self.backbone.load_state_dict(checkpoint["models"], strict=False)
        statistic_names = ("history_mass", "history_mean", "history_m2",
                           "pending_mass", "pending_mean", "pending_m2", "processed_count")
        missing_parameters = [key for key in loaded.missing_keys
                              if not any(key.split(".")[-1].startswith(n) for n in statistic_names)]
        if missing_parameters:
            raise ValueError(f"Checkpoint lacks required SPOL tensors: {missing_parameters[:8]}")
        if update_bank and loaded.missing_keys:
            raise ValueError("Online updates require current SPOL sufficient statistics; "
                             f"missing tensors: {loaded.missing_keys[:8]}")
        # The runtime bank now uses initial EM followed by responsibility-based online EM.
        self.gmm_update_rule = "online_em"
        self.num_cls = self.backbone.num_cls
        self.update_bank = update_bank
        # The training engine enables online updates from the first Horizon epoch.
        self.prototype_update_enabled = False
        self.loaded_epoch = checkpoint.get("epoch")

        self.select_bank(bank_group, bank_size, num_components, threshold)

    @property
    def PrototypeSupportBank(self):
        """Reference the loaded bank without duplicate module registration."""
        return getattr(self.backbone, self.selected_bank_name)

    @property
    def num_prototype(self):
        """Actual GMM samples or saved OT centers per class for the selected bank."""
        return self.PrototypeSupportBank.num_prototypes

    def cam_masks(self, cam, label=None):
        """Constrain first-view CAM masks with available image-level labels."""
        normalized = F.relu(cam.detach())
        normalized = normalized / (F.adaptive_max_pool2d(normalized, 1) + 1e-5)
        if label is not None:
            labels = label.detach().to(device=cam.device, dtype=cam.dtype).reshape(-1, 1)
            if labels.shape[0] != cam.shape[0] or not ((labels == 0) | (labels == 1)).all():
                raise ValueError("Expected one binary label per image")
            normalized = normalized * labels.unsqueeze(-1).unsqueeze(-1)
        return normalized > self.threshold, normalized > 0

    def selected_bank_status(self):
        bank = self.PrototypeSupportBank
        if self.bank_group == "ot":
            status = dict(counts=[int(bank.back_count), int(bank.fore_count)],
                          fits=bank.num_updates.tolist())
        else:
            status = dict(counts=[int(getattr(bank, f"queue_count{i}")) for i in range(2)],
                          fits=[int(getattr(bank, f"fit_count{i}")) for i in range(2)],
                          num_components=bank.num_components,
                          foreground_components=bank.foreground_components, threshold=self.threshold)
        status.update(name=self.selected_bank_name, group=self.bank_group,
                      bank_size=bank.bank_size,
                      feature_dim=self.backbone.feature_dim,
                      num_prototypes=self.num_prototype,
                      ready=all(count > 0 for count in status["fits"]))
        if self.bank_group != "ot":
            status.update(covariance_type=bank.covariance_type,
                          update_rule=self.gmm_update_rule,
                          init_iterations=bank.init_iterations,
                          covariance_shrinkage=bank.covariance_shrinkage,
                          gamma=bank.momentum_gamma if self.gmm_update_rule in ("alpha_gamma", "gamma", "online_em", "matched_em") else None)
        return status

    def require_bank_ready(self):
        bank = self.PrototypeSupportBank
        ready = (bool((bank.num_updates > 0).all()) if self.bank_group == "ot" else
                 all(int(getattr(bank, f"fit_count{i}")) > 0 for i in range(2)))
        if not ready:
            status = self.selected_bank_status()
            raise RuntimeError(
                f"Selected bank {status['name']} is not fitted for both classes: "
                f"counts={status['counts']}, updates={status['fits']}. "
                "Choose a checkpoint saved after both classes have initialized; "
                "warmup checkpoints cannot produce prototype similarity maps.")

    def select_bank(self, bank_group="components", bank_size=1000,
                    num_components=3, threshold=0.8):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        if bank_group == "ot":
            if self.update_bank:
                raise ValueError("Horizon uses fixed OT centers; update OT through SPOL training")
            names = [name for name, size in self.backbone.cluster_bank_configs.items()
                     if size == bank_size]
        elif bank_group in ("components", "capacity", "threshold"):
            names = [name for name, (t, size, components) in self.backbone.bank_configs.items()
                     if self.backbone.bank_groups[name] == bank_group and size == bank_size
                     and components == num_components and abs(t - threshold) < 1e-8]
        else:
            raise ValueError("bank_group must be components, capacity, threshold or ot")
        if len(names) != 1:
            raise ValueError(f"Expected one bank for group={bank_group}, size={bank_size}, "
                             f"components={num_components}, threshold={threshold}; found {names}")
        name = names[0]
        bank = getattr(self.backbone, name)
        if bank_group != "ot" and bank.num_components != bank.foreground_components:
            raise ValueError("Horizon PPC requires equal foreground/background component counts")
        self.selected_bank_name = name
        self.bank_group, self.bank_size = bank_group, bank_size
        self.threshold = threshold
        return bank

    def cam_phase(self, x, label=None, skull_mask=None):
        scores, cam, hie_fea, _ = self.backbone.forward_features(x)
        if not (self.training and self.update_bank and self.prototype_update_enabled and label is not None):
            return scores, cam, hie_fea
        devices = [x.device.index] if x.is_cuda else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            norm_cam = F.relu(cam.detach())
            norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, 1) + 1e-5)
            labels = label.detach().to(cam.device).reshape(-1)
            if labels.numel() != cam.shape[0] or not ((labels == 0) | (labels == 1)).all():
                raise ValueError("Expected one binary label per image")
            positive = (labels == 1)[:, None, None, None]
            sampling_mask = dilated_skull_mask(skull_mask, norm_cam)
            pseudo_label = torch.full_like(norm_cam, -1, dtype=torch.long)
            pseudo_label[(norm_cam == 0) & positive & sampling_mask] = 0
            pseudo_label[(norm_cam > self.threshold) & positive & sampling_mask] = 1
            # Preserve the spatial map: current SPOL samples a foreground point and
            # a normalized local-background mean per eligible image.
            samples = PrototypeSupportBank.select_samples(pseudo_label, hie_fea)
            self.PrototypeSupportBank.update_samples(samples)
        return scores, cam, hie_fea

    def proto_phase(self, hie_fea, spatial_size):
        self.require_bank_ready()
        n = hie_fea.shape[0]
        h, w = spatial_size
        if hie_fea.ndim != 3 or hie_fea.shape[1:] != (h * w, self.backbone.feature_dim):
            raise ValueError("Hierarchical feature shape must match the SPOL spatial size and feature dimension")
        proto_fea = l2_normalize(hie_fea).reshape(n * h * w, hie_fea.size(-1))
        bank = self.PrototypeSupportBank
        if self.bank_group == "ot":
            self.ppc_options = dict(getattr(self, "ppc_options", {}),
                                    num_components=bank.context_prototypes.shape[1])
            centers = l2_normalize(bank.context_prototypes.detach()).to(hie_fea.device)
            proto_logits = torch.einsum("nd,ckd->nck", proto_fea, centers)
        else:
            self.ppc_options = dict(getattr(self, "ppc_options", {}),
                                    num_components=bank.gmm_means_0.shape[0])
            sampled_protos = bank.sample_prototypes().to(hie_fea.device).detach()
            sampled_protos = l2_normalize(sampled_protos)
            proto_logits = torch.einsum("nd,ckd->nck", proto_fea, sampled_protos)
            global_proto = torch.stack([bank.gmm_means_0, bank.gmm_means_1], dim=0).detach()
            # Append K centers; PPC uses background max and foreground mean over them.
            global_proto = l2_normalize(global_proto).to(hie_fea.device)
            global_logits = torch.einsum("nd,ckd->nck", proto_fea, global_proto)
            proto_logits = torch.cat([proto_logits, global_logits], dim=-1)
        similarities = proto_logits.amax(-1)
        sim = rearrange(similarities, "(b h w) c -> b c h w", b=n, h=h, w=w)
        all_logits = rearrange(proto_logits, "(b h w) c k -> b c k h w", b=n, h=h, w=w)
        proto_fore_mask = (sim[:, 1:] - sim[:, :1]) > 0
        return proto_logits, sim, all_logits, proto_fore_mask

    def forward(self, x, label=None, skull_mask=None):
        scores, cam, hie_fea = self.cam_phase(x, label, skull_mask)
        _, sim, all_logits, _ = self.proto_phase(hie_fea, cam.shape[-2:])
        return scores, cam, sim, hie_fea


@register_model
def build_plug(pretrained=True, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    return Plug_play(**kwargs)
