# requirements_patch.sh — run once after pip install
# Fixes: zip(strict=True) incompatibility between PyTorch 2.11 and DeepSpeed 0.18.1
TORCH_SCHED=$(python -c "import torch.optim.lr_scheduler as s; import inspect; print(inspect.getfile(s))")
sed -i 's/zip(self.optimizer.param_groups, values, strict=True)/zip(self.optimizer.param_groups, values)/' $TORCH_SCHED
echo "Patch applied to $TORCH_SCHED"