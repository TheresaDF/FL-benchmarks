import torch
from src.client.fedavg import FedAvgClient


class pFLMFClient(FedAvgClient):
    """
    pFLMF client alternating between v-steps (personal params) and u-steps (shared params).
    Compatible with FL-bench model definitions (including lowrank_dnn*).
    """

    def __init__(self, **commons):
        super().__init__(**commons)

        pflmf_args = getattr(self.args, "pflmf", None)
        self.v_steps = int(getattr(pflmf_args, "num_steps_v", 1)) if pflmf_args else 1
        self.u_steps = int(getattr(pflmf_args, "num_steps_u", 1)) if pflmf_args else 1
        self.lr_v    = float(getattr(pflmf_args, "lr_v", 1e-4))   if pflmf_args else 1e-4
        self.lr_u    = float(getattr(pflmf_args, "lr_u", 1e-3))   if pflmf_args else 1e-3

        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]

        # ---- Handle models that define classifier as a Parameter instead of a Module ----
        names = set(getattr(self, "personal_params_name", []))
        if not names and hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, torch.nn.Module):
                for n, _ in self.model.classifier.named_parameters():
                    names.add(f"classifier.{n}")
            elif isinstance(self.model.classifier, torch.nn.Parameter):
                # It’s a single Parameter, so just record its own name
                names.add("classifier")
        self._personal_names = names

    # -----------------------------------------------------------------
    def _all_param_names(self):
        return {n for n, _ in self.model.named_parameters()}

    def _params_from_names(self, names_set):
        update, freeze = [], []
        for n, p in self.model.named_parameters():
            (update if n in names_set else freeze).append(p)
        return update, freeze

    def _set_requires_grad_mask(self, active):
        active = set(active)
        for p in self.model.parameters():
            p.requires_grad = p in active

    def _set_optimizer_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _phase(self, target_names, steps, lr):
        if steps <= 0 or lr <= 0:
            return
        update, freeze = self._params_from_names(target_names)
        self._set_requires_grad_mask(update)
        self._set_optimizer_lr(lr)
        self.model.train()
        for _ in range(steps):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for p in freeze:
                    if p.grad is not None:
                        p.grad = None
                self.optimizer.step()
        for p in self.model.parameters():
            p.requires_grad = True
            if p.grad is not None:
                p.grad = None

    def fit(self):
        # v-steps: personal
        self._phase(self._personal_names, self.v_steps, self.lr_v)
        # u-steps: shared
        shared = self._all_param_names() - self._personal_names
        self._phase(shared, self.u_steps, self.lr_u)
        for g, lr in zip(self.optimizer.param_groups, self._base_lrs):
            g["lr"] = lr
        return {}
