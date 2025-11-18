import os
from pathlib import Path

from ipdb import set_trace as bp

import math
import random
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from src.behavior.diffusion import DiffusionPolicy
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from src.behavior.residual_mlp import ResidualMlpPolicy
from src.eval.eval_utils import get_model_from_api_or_cached
from diffusers.optimization import get_scheduler


from src.gym.env_rl_wrapper import RLPolicyEnvWrapper
from src.common.config_util import merge_base_bc_config_with_root_config
from src.gym.observation import DEFAULT_STATE_OBS

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import trange, tqdm

import wandb
from wandb.apis.public.runs import Run
from wandb.errors.util import CommError

from src.gym import get_rl_env
import gymnasium as gym

# TRAK imports
try:
    import trak
    from trak.modelout_functions import AbstractModelOutput
    from trak.gradient_computers import AbstractGradientComputer
    TRAK_AVAILABLE = True
except ImportError:
    TRAK_AVAILABLE = False
    print("Warning: TRAK not available. Install with: pip install traker")

# Register the eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)


@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    steps_per_iteration: int,
    discount: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + discount * gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns


class PPOModelOutput(AbstractModelOutput):
    """PPO loss for TRAK influence computation."""
    
    def __init__(self, clip_coef: float = 0.2, vf_coef: float = 1.0):
        super().__init__()
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
    
    def get_output(
        self,
        model,
        weights: dict,
        buffers: dict,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PPO loss for a single sample."""
        # Unsqueeze to add batch dimension (for single sample)
        obs = obs.unsqueeze(0)
        actions = actions.unsqueeze(0)
        
        # Separate weights for actor_mean and critic
        actor_weights = {k.replace("actor_mean.", ""): v for k, v in weights.items() if k.startswith("actor_mean.")}
        critic_weights = {k.replace("critic.", ""): v for k, v in weights.items() if k.startswith("critic.")}
        logstd = weights["actor_logstd"]
        
        # Forward pass through actor_mean using functional_call
        action_mean = torch.func.functional_call(
            model.actor_mean, 
            actor_weights, 
            obs
        )
        
        # Compute log probability manually (avoid Normal distribution for vmap compatibility)
        action_logstd = logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Gaussian log probability: log p(x|μ,σ) = -0.5*log(2π) - log(σ) - 0.5*((x-μ)/σ)²
        log_2pi = math.log(2 * math.pi)
        var = action_std ** 2
        log_prob = -0.5 * (((actions - action_mean) ** 2) / var + torch.log(var) + log_2pi)
        new_logprob = log_prob.sum(dim=1)
        
        # Forward pass through critic
        newvalue = torch.func.functional_call(
            model.critic,
            critic_weights,
            obs
        )
        
        # PPO clipped policy loss
        logratio = new_logprob - old_logprobs
        ratio = logratio.exp()
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Value loss
        v_loss = 0.5 * ((newvalue.squeeze() - returns) ** 2).mean()
        
        # Total loss
        loss = pg_loss + self.vf_coef * v_loss
        return loss
    
    def get_out_to_loss_grad(self, model, weights, buffers, batch) -> torch.Tensor:
        """Gradient of loss w.r.t. model output (always 1 for direct loss)."""
        # batch is a dict, get device from any tensor
        device = batch["obs"].device if "obs" in batch else next(iter(batch.values())).device
        return torch.ones(1).to(device)


class PPOGradientComputer(AbstractGradientComputer):
    """Gradient computer for PPO policy."""
    
    def __init__(
        self,
        model,
        task: PPOModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt=None,  # TRAK passes this, we ignore it for now (compute gradients for all params)
    ):
        super().__init__(model, task, grad_dim, dtype, device)
        self.grad_wrt = grad_wrt  # Store but don't use (compute full gradients)
        self.load_model_params(model)
    
    def load_model_params(self, model):
        """Load model parameters."""
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())
        
        # Verify gradient dimension
        total_params = sum(p.numel() for p in self.func_weights.values())
        assert self.grad_dim == total_params, f"grad_dim mismatch: {self.grad_dim} vs {total_params}"
    
    def compute_per_sample_grad(self, batch: dict) -> torch.Tensor:
        """Compute per-sample gradients using torch.func."""
        # Extract batch elements
        obs = batch["obs"]
        actions = batch["actions"]
        old_logprobs = batch["old_logprobs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        
        batch_size = obs.shape[0]
        
        # Compute gradients w.r.t. weights (argnums=1)
        grads_loss = torch.func.grad(
            self.modelout_fn.get_output, has_aux=False, argnums=1
        )
        
        # Vectorize over batch dimension
        grads = torch.func.vmap(
            grads_loss,
            in_dims=(None, None, None, 0, 0, 0, 0, 0),
            randomness="different",
        )(
            self.model,
            self.func_weights,
            self.func_buffers,
            obs,
            actions,
            old_logprobs,
            advantages,
            returns,
        )
        
        # Flatten gradients to vector form
        grads_flat = torch.zeros(
            size=(batch_size, self.grad_dim),
            dtype=obs.dtype,
            device=obs.device,
        )
        
        pointer = 0
        for param_name in sorted(self.func_weights.keys()):
            param_grads = grads[param_name]
            num_param = param_grads[0].numel()
            grads_flat[:, pointer:pointer + num_param] = param_grads.flatten(start_dim=1)
            pointer += num_param
        
        return grads_flat
    
    def compute_loss_grad(self, batch: dict) -> torch.Tensor:
        """Gradient of loss w.r.t. model output."""
        return self.modelout_fn.get_out_to_loss_grad(
            self.model, self.func_weights, self.func_buffers, batch
        )


def compute_cupid_trak_influence(
    policy,
    train_data: dict,
    test_data: dict,
    device: torch.device,
    iteration: int,
    clip_coef: float = 0.2,
    vf_coef: float = 1.0,
    proj_dim: int = 512,
    use_half_precision: bool = True,
) -> np.ndarray:
    """
    Compute TRAK influence scores for PPO.
    
    Args:
        policy: The residual policy model
        train_data: Dict with keys: obs, actions, old_logprobs, advantages, returns
        test_data: Dict with keys: obs, actions, old_logprobs, advantages, returns (successful samples)
        device: Device to run computation on
        iteration: Current iteration number (for cache directory naming)
        clip_coef: PPO clipping coefficient
        vf_coef: Value function loss coefficient
        proj_dim: TRAK projection dimension
        use_half_precision: Whether to use half precision
    
    Returns:
        Influence matrix of shape (train_size, test_size)
    """
    if not TRAK_AVAILABLE:
        raise ImportError("TRAK not available. Install with: pip install traker")
    
    import pathlib
    
    train_size = train_data["obs"].shape[0]
    test_size = test_data["obs"].shape[0]
    
    print(f"Computing TRAK influence: {train_size} train samples, {test_size} test samples")
    
    # Compute gradient dimension
    grad_dim = sum(p.numel() for p in policy.parameters())
    print(f"Gradient dimension: {grad_dim}")
    
    # Create task and gradient computer
    task = PPOModelOutput(clip_coef=clip_coef, vf_coef=vf_coef)
    gradient_computer = PPOGradientComputer
    
    # Create cache directory for TRAK
    # Use iteration + train_size + test_size to make it unique
    # This prevents cache collisions when subsampling varies
    save_dir = pathlib.Path("./trak_cache") / f"iter_{iteration}_train{train_size}_test{test_size}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"TRAK cache directory: {save_dir}")
    
    # Create TRAKer
    traker = trak.TRAKer(
        model=policy,
        task=task,
        train_set_size=train_size,
        gradient_computer=gradient_computer,
        device=device,
        proj_dim=min(proj_dim, grad_dim),
        save_dir=str(save_dir),  # ✅ Fixed: provide valid directory path
        use_half_precision=use_half_precision,
    )
    
    # Load checkpoint
    traker.load_checkpoint(policy.state_dict(), model_id=0)
    
    # Featurize training data
    train_batch_size = min(256, train_size)
    for start_idx in tqdm(range(0, train_size, train_batch_size), desc="Featurizing train"):
        end_idx = min(start_idx + train_batch_size, train_size)
        batch = {
            "obs": train_data["obs"][start_idx:end_idx].to(device),
            "actions": train_data["actions"][start_idx:end_idx].to(device),
            "old_logprobs": train_data["old_logprobs"][start_idx:end_idx].to(device),
            "advantages": train_data["advantages"][start_idx:end_idx].to(device),
            "returns": train_data["returns"][start_idx:end_idx].to(device),
        }
        traker.featurize(batch, num_samples=end_idx - start_idx)
    
    # Finalize features
    traker.finalize_features(model_ids=[0])
    
    # Score test data
    traker.start_scoring_checkpoint(
        exp_name="test",
        checkpoint=policy.state_dict(),
        model_id=0,
        num_targets=test_size,
    )
    
    test_batch_size = min(256, test_size)
    for start_idx in tqdm(range(0, test_size, test_batch_size), desc="Scoring test"):
        end_idx = min(start_idx + test_batch_size, test_size)
        batch = {
            "obs": test_data["obs"][start_idx:end_idx].to(device),
            "actions": test_data["actions"][start_idx:end_idx].to(device),
            "old_logprobs": test_data["old_logprobs"][start_idx:end_idx].to(device),
            "advantages": test_data["advantages"][start_idx:end_idx].to(device),
            "returns": test_data["returns"][start_idx:end_idx].to(device),
        }
        traker.score(batch=batch, num_samples=end_idx - start_idx)
    
    # Finalize scores
    # Note: This can be memory-intensive for large datasets
    # TRAK tries to allocate train_size * test_size * some_factor in memory
    print(f"Finalizing scores (may require significant memory)...")
    print(f"Expected influence matrix size: {train_size} x {test_size}")
    expected_memory_gb = (train_size * test_size * 4) / (1024**3)  # fp32 estimate
    if use_half_precision:
        expected_memory_gb /= 2
    print(f"Estimated memory requirement: ~{expected_memory_gb:.1f} GB")
    
    try:
        scores = traker.finalize_scores(exp_name="test", model_ids=[0])
        scores = np.array(scores)  # Shape: (train_size, test_size)
    except RuntimeError as e:
        if "allocate memory" in str(e):
            print(f"ERROR: Out of memory during score finalization")
            print(f"Try reducing: cupid_max_train_samples (data collection subsample size)")
            print(f"             proj_dim (current: {proj_dim})")
            print(f"Current sizes: {train_size} train, {test_size} test")
            raise
        else:
            raise
    
    return scores


@hydra.main(
    config_path="../config",
    config_name="base_residual_rl",
    version_base="1.2",
)
def main(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)

    if (job_id := os.environ.get("SLURM_JOB_ID")) is not None:
        cfg.slurm_job_id = job_id

    # Ensure exactly one of cfg.base_policy.wandb_id or cfg.base_policy.wt_path is set
    assert (
        sum(
            [
                cfg.base_policy.wandb_id is not None,
                cfg.base_policy.wt_path is not None,
            ]
        )
        == 1
    ), "Exactly one of base_policy.wandb_id or base_policy.wt_path must be set"

    run_state_dict = None

    # Check if we are continuing a run
    run_exists = False
    if cfg.wandb.continue_run_id is not None:
        try:
            run: Run = wandb.Api().run(
                f"{cfg.wandb.project}/{cfg.wandb.continue_run_id}"
            )
            run_exists = True
        except (ValueError, CommError):
            pass

    if run_exists:
        print(f"Continuing run {cfg.wandb.continue_run_id}, {run.name}")

        run_id = cfg.wandb.continue_run_id
        run_path = f"{cfg.wandb.project}/{run_id}"

        # Load the weights from the run
        cfg, wts = get_model_from_api_or_cached(
            run_path, "latest", wandb_mode=cfg.wandb.mode
        )

        # Update the cfg.continue_run_id to the run_id
        cfg.wandb.continue_run_id = run_id

        base_cfg = cfg.base_policy
        merge_base_bc_config_with_root_config(cfg, base_cfg)

        print(f"Loading weights from {wts}")

        run_state_dict = torch.load(wts)

        # Set the best test loss and success rate to the one from the run
        try:
            best_eval_success_rate = run.summary["eval/best_eval_success_rate"]
        except KeyError:
            best_eval_success_rate = run.summary["eval/success_rate"]

        iteration = run.summary["iteration"]
        global_step = run.lastHistoryStep
        sps = run.summary.get("charts/SPS", run.summary.get("training/SPS", 0))
        training_cum_time = sps * global_step
        run_name = run.name

    else:
        global_step = 0
        iteration = 0
        best_eval_success_rate = 0.0
        training_cum_time = 0

        # Load the behavior cloning actor
        if cfg.base_policy.wandb_id is not None:
            base_cfg, base_wts = get_model_from_api_or_cached(
                cfg.base_policy.wandb_id,
                wt_type=cfg.base_policy.wt_type,
                wandb_mode=cfg.wandb.mode,
            )
        elif cfg.base_policy.wt_path is not None:
            base_wts = cfg.base_policy.wt_path
            base_cfg: DictConfig = OmegaConf.create(torch.load(base_wts)["config"])
        else:
            raise ValueError("No base policy provided")

        merge_base_bc_config_with_root_config(cfg, base_cfg)
        cfg.actor_name = f"residual_{cfg.base_policy.actor.name}"

        if cfg.seed is None:
            cfg.seed = random.randint(0, 2**32 - 1)

        run_name = f"{int(time.time())}__{cfg.actor_name}_ppo__{cfg.seed}"

    if "task" not in cfg.env:
        cfg.env.task = "one_leg"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    gpu_id = cfg.gpu_id
    device = torch.device(f"cuda:{gpu_id}")

    env: gym.Env = get_rl_env(
        gpu_id=gpu_id,
        act_rot_repr=cfg.control.act_rot_repr,
        action_type=cfg.control.control_mode,
        april_tags=False,
        concat_robot_state=True,
        ctrl_mode=cfg.control.controller,
        obs_keys=DEFAULT_STATE_OBS,
        task=cfg.env.task,
        compute_device_id=gpu_id,
        graphics_device_id=gpu_id,
        headless=cfg.headless,
        num_envs=cfg.num_envs,
        observation_space="state",
        randomness=cfg.env.randomness,
        max_env_steps=100_000_000,
    )

    n_parts_to_assemble = env.n_parts_assemble

    if cfg.base_policy.actor.name == "diffusion":
        agent = ResidualDiffusionPolicy(device, base_cfg)
    elif cfg.base_policy.actor.name == "mlp":
        agent = ResidualMlpPolicy(device, base_cfg)
    else:
        raise ValueError(f"Unknown actor type: {cfg.base_policy.actor}")

    agent.to(device)
    agent.eval()

    # Set the inference steps of the actor
    if isinstance(agent, DiffusionPolicy):
        agent.inference_steps = 4

    env: RLPolicyEnvWrapper = RLPolicyEnvWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        normalize_reward=cfg.normalize_reward,
        reset_on_success=cfg.reset_on_success,
        reset_on_failure=cfg.reset_on_failure,
        reward_clip=cfg.clip_reward,
        sample_perturbations=cfg.sample_perturbations,
        device=device,
    )

    optimizer_actor = optim.AdamW(
        agent.actor_parameters,
        lr=cfg.learning_rate_actor,
        betas=cfg.get("optimizer_betas_actor", (0.9, 0.999)),
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_actor = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_actor,
        num_warmup_steps=cfg.lr_scheduler.actor_warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    optimizer_critic = optim.AdamW(
        agent.critic_parameters,
        lr=cfg.learning_rate_critic,
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_critic = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_critic,
        num_warmup_steps=cfg.lr_scheduler.critic_warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    if run_state_dict is not None:
        if "actor_logstd" in run_state_dict["model_state_dict"]:
            agent.residual_policy.load_state_dict(run_state_dict["model_state_dict"])
        else:
            agent.load_state_dict(run_state_dict["model_state_dict"])

        optimizer_actor.load_state_dict(run_state_dict["optimizer_actor_state_dict"])
        optimizer_critic.load_state_dict(run_state_dict["optimizer_critic_state_dict"])
        lr_scheduler_actor.load_state_dict(run_state_dict["scheduler_actor_state_dict"])
        lr_scheduler_critic.load_state_dict(
            run_state_dict["scheduler_critic_state_dict"]
        )
    else:
        agent.load_base_state_dict(base_wts)

    residual_policy = agent.residual_policy

    if (
        "pretrained_wts" in cfg.actor.residual_policy
        and cfg.actor.residual_policy.pretrained_wts
    ):
        print(
            f"Loading pretrained weights from {cfg.actor.residual_policy.pretrained_wts}"
        )
        run_state_dict = torch.load(cfg.actor.residual_policy.pretrained_wts)

        if "actor_logstd" in run_state_dict["model_state_dict"]:
            agent.residual_policy.load_state_dict(run_state_dict["model_state_dict"])
        else:
            agent.load_state_dict(run_state_dict["model_state_dict"])
        optimizer_actor.load_state_dict(run_state_dict["optimizer_actor_state_dict"])
        optimizer_critic.load_state_dict(run_state_dict["optimizer_critic_state_dict"])
        lr_scheduler_actor.load_state_dict(run_state_dict["scheduler_actor_state_dict"])
        lr_scheduler_critic.load_state_dict(
            run_state_dict["scheduler_critic_state_dict"]
        )

    steps_per_iteration = cfg.data_collection_steps

    print(f"Total timesteps: {cfg.total_timesteps}, batch size: {cfg.batch_size}")
    print(
        f"Mini-batch size: {cfg.minibatch_size}, num iterations: {cfg.num_iterations}"
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))

    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        save_code=True,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
    )

    obs: torch.Tensor = torch.zeros(
        (
            steps_per_iteration,
            cfg.num_envs,
            residual_policy.obs_dim,
        )
    )
    actions = torch.zeros((steps_per_iteration, cfg.num_envs) + env.action_space.shape)
    logprobs = torch.zeros((steps_per_iteration, cfg.num_envs))
    rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    dones = torch.zeros((steps_per_iteration, cfg.num_envs))
    values = torch.zeros((steps_per_iteration, cfg.num_envs))

    start_time = time.time()

    next_done = torch.zeros(cfg.num_envs)
    next_obs = env.reset()
    agent.reset()

    # Create model save dir
    model_save_dir: Path = Path("models") / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # CUPID: Initialize variables for TRAK-based sample importance weighting
    use_cupid = cfg.get("use_cupid", False)
    cupid_compute_interval = cfg.get("cupid_compute_interval", 1)  # Default: compute every iteration
    cached_sample_importance = None
    
    if use_cupid:
        if not TRAK_AVAILABLE:
            print("WARNING: CUPID enabled but TRAK not available!")
            print("Install with: pip install traker")
            print("Falling back to uniform sampling...")
            use_cupid = False
        else:
            print(f"CUPID (TRAK) settings:")
            print(f"  - Compute interval: {cupid_compute_interval}")
            print(f"  - Projection dim: {cfg.get('cupid_proj_dim', 512)}")
            print(f"  - Half precision: {cfg.get('cupid_half_precision', True)}")
            print(f"  - Max train samples (data subsample): {cfg.get('cupid_max_train_samples', 100000)}")
            print(f"  - PPO trains ONLY on subsampled data with CUPID weights")

    while global_step < cfg.total_timesteps:
        iteration += 1
        print(f"Iteration: {iteration}/{cfg.num_iterations}")
        print(f"Run name: {run_name}")
        iteration_start_time = time.time()

        # If eval first flag is set, we will evaluate the model before doing any training
        eval_mode = (iteration - int(cfg.eval_first)) % cfg.eval_interval == 0

        # Also reset the env to have more consistent results
        if eval_mode or cfg.reset_every_iteration:
            next_obs = env.reset()
            agent.reset()

        print(f"Eval mode: {eval_mode}")

        for step in range(0, steps_per_iteration):
            if not eval_mode:
                # Only count environment steps during training
                global_step += cfg.num_envs

            # Get the base normalized action
            base_naction = agent.base_action_normalized(next_obs)

            # Process the obs for the residual policy
            next_nobs = agent.process_obs(next_obs)
            next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)

            dones[step] = next_done
            obs[step] = next_residual_nobs

            with torch.no_grad():
                residual_naction_samp, logprob, _, value, naction_mean = (
                    residual_policy.get_action_and_value(next_residual_nobs)
                )

            residual_naction = residual_naction_samp if not eval_mode else naction_mean
            naction = base_naction + residual_naction * residual_policy.action_scale

            action = agent.normalizer(naction, "action", forward=False)
            next_obs, reward, next_done, truncated, info = env.step(action)

            if cfg.truncation_as_done:
                next_done = next_done | truncated

            values[step] = value.flatten().cpu()
            actions[step] = residual_naction.cpu()
            logprobs[step] = logprob.cpu()
            rewards[step] = reward.view(-1).cpu()
            next_done = next_done.view(-1).cpu()

            if step > 0 and (env_step := step * 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()} fps={env_step * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
                )

        # Calculate the success rate
        # Find the rewards that are not zero
        # Env is successful if it received a reward more than or equal to n_parts_to_assemble
        env_success = (rewards > 0).sum(dim=0) >= n_parts_to_assemble
        success_rate = env_success.float().mean().item()

        if success_rate > 0:
            # Calculate the share of timesteps that come from successful trajectories that account for the success rate and the varying number of timesteps per trajectory
            # Count total timesteps in successful trajectories
            timesteps_in_success = rewards[:, env_success]

            # Find index of last reward in each trajectory
            # This has all timesteps including and after episode is done
            success_dones = timesteps_in_success.cumsum(dim=0) >= n_parts_to_assemble
            last_reward_idx = success_dones.int().argmax(dim=0)

            # Calculate the total number of timesteps in successful trajectories
            total_timesteps_in_success = (last_reward_idx + 1).sum().item()

            # Calculate the share of successful timesteps
            success_timesteps_share = total_timesteps_in_success / rewards.numel()

            # Mean successful episode length
            mean_success_episode_length = (
                total_timesteps_in_success / env_success.sum().item()
            )
            max_success_episode_length = last_reward_idx.max().item()
        else:
            success_timesteps_share = 0
            mean_success_episode_length = 0
            max_success_episode_length = 0

        print(
            f"SR: {success_rate:.4%}, SPS: {steps_per_iteration * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
            f", STS: {success_timesteps_share:.4%}, MSEL: {mean_success_episode_length:.2f}"
        )

        if eval_mode:
            # If we are in eval mode, we don't need to do any training, so log the result and continue

            # Save the model if the evaluation success rate improves
            if success_rate > best_eval_success_rate:
                best_eval_success_rate = success_rate
                model_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")
                torch.save(
                    {
                        # Save the weights of the residual policy (base + residual)
                        "model_state_dict": agent.state_dict(),
                        "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                        "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                        "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                        "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "success_rate": success_rate,
                        "success_timesteps_share": success_timesteps_share,
                        "iteration": iteration,
                        "training_cum_time": training_cum_time,
                    },
                    model_path,
                )

                wandb.save(model_path)
                print(f"Evaluation success rate improved. Model saved to {model_path}")

            wandb.log(
                {
                    "eval/success_rate": success_rate,
                    "eval/best_eval_success_rate": best_eval_success_rate,
                    "iteration": iteration,
                },
                step=global_step,
            )
            # Start the data collection again
            # NOTE: We're not resetting here now, that happens before the next
            # iteration only if the reset_every_iteration flag is set
            continue

        b_obs = obs.reshape((-1, residual_policy.obs_dim))
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)

        # Get the base normalized action
        # Process the obs for the residual policy
        base_naction = agent.base_action_normalized(next_obs)
        next_nobs = agent.process_obs(next_obs)
        next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)
        next_value = residual_policy.get_value(next_residual_nobs).reshape(1, -1).cpu()

        # bootstrap value if not done
        advantages, returns = calculate_advantage(
            values,
            next_value,
            rewards,
            dones,
            next_done,
            steps_per_iteration,
            cfg.discount,
            cfg.gae_lambda,
        )

        b_advantages = advantages.reshape(-1).cpu()
        b_returns = returns.reshape(-1).cpu()
        
        # CUPID: Subsample collected data BEFORE TRAK computation
        # This ensures TRAK and PPO work on the SAME dataset
        original_batch_size = b_obs.shape[0]
        subsample_indices = None  # Track which original indices we keep
        
        if use_cupid and original_batch_size > cfg.get("cupid_max_train_samples", 100000):
            print(f"\n{'='*60}")
            print(f"Subsampling data for CUPID: {original_batch_size} → {cfg.get('cupid_max_train_samples', 100000)}")
            print(f"{'='*60}")
            
            # Random subsample
            subsample_indices = np.random.choice(
                original_batch_size, 
                cfg.get("cupid_max_train_samples", 100000), 
                replace=False
            )
            subsample_indices = np.sort(subsample_indices)  # Keep temporal order
            
            # Subsample all data
            b_obs = b_obs[subsample_indices]
            b_actions = b_actions[subsample_indices]
            b_logprobs = b_logprobs[subsample_indices]
            b_values = b_values[subsample_indices]
            b_advantages = b_advantages[subsample_indices]
            b_returns = b_returns[subsample_indices]
            
            print(f"Using {b_obs.shape[0]} samples for TRAK computation and PPO training")
            print(f"Subsampling ratio: {b_obs.shape[0] / original_batch_size:.2%}")
            print(f"{'='*60}\n")

        # CUPID: Compute sample importance scores using TRAK
        # NOTE: Must compute every iteration since data changes each iteration!
        # The cupid_compute_interval controls whether to compute or use uniform sampling.
        cached_sample_importance = None  # Reset each iteration
        
        if use_cupid and TRAK_AVAILABLE and (iteration % cupid_compute_interval == 0):
            print(f"\n{'='*60}")
            print(f"Computing CUPID (TRAK) influence scores at iteration {iteration}...")
            print(f"{'='*60}")
            cupid_start_time = time.time()
            
            # Prepare training data (all collected samples)
            train_data = {
                "obs": b_obs,
                "actions": b_actions,
                "old_logprobs": b_logprobs,
                "advantages": b_advantages,
                "returns": b_returns,
            }
            
            # Prepare test data (only successful samples)
            # Flatten and select successful trajectory samples
            flat_rewards = rewards.reshape(-1)
            flat_env_success = env_success.repeat_interleave(steps_per_iteration)
            success_mask = flat_env_success.cpu().numpy()
            
            # IMPORTANT: If we subsampled the data, apply same indices to success_mask
            if subsample_indices is not None:
                success_mask = success_mask[subsample_indices]
            
            if success_mask.sum() > 0:
                test_data = {
                    "obs": b_obs[success_mask],
                    "actions": b_actions[success_mask],
                    "old_logprobs": b_logprobs[success_mask],
                    "advantages": b_advantages[success_mask],
                    "returns": b_returns[success_mask],
                }
                
                print(f"Train size: {train_data['obs'].shape[0]}")
                print(f"Test size (successful): {test_data['obs'].shape[0]}")
                
                # Compute TRAK influence
                influence_scores = compute_cupid_trak_influence(
                    policy=residual_policy,
                    train_data=train_data,
                    test_data=test_data,
                    device=device,
                    iteration=iteration,
                    clip_coef=cfg.clip_coef,
                    vf_coef=cfg.vf_coef,
                    proj_dim=cfg.get("cupid_proj_dim", 512),
                    use_half_precision=cfg.get("cupid_half_precision", True),
                )
                
                # Aggregate influence scores: weighted sum over test samples
                # CUPID's Performance Influence: Ψπ-inf(ξ) = E[R(τ)/H · Ψa-inf(ξ', ξ)]
                # Weight test samples by their returns (higher return = more important)
                test_returns = b_returns[success_mask].cpu().numpy()  # Returns for successful test samples
                test_weights = test_returns / (test_returns.sum() + 1e-8)  # Normalize to sum to 1
                
                # Weighted aggregation: sum over test samples weighted by their returns
                # influence_scores shape: (train_size, test_size)
                # test_weights shape: (test_size,)
                # Result shape: (train_size,)
                cached_sample_importance = (influence_scores * test_weights[np.newaxis, :]).sum(axis=1)
                
                # Normalize to probabilities for sampling
                # Shift to make all non-negative (if there are negative influences)
                min_score = cached_sample_importance.min()
                if min_score < 0:
                    cached_sample_importance = cached_sample_importance - min_score
                
                # Normalize to sum to 1
                cached_sample_importance = cached_sample_importance / (cached_sample_importance.sum() + 1e-8)
                
                cupid_compute_time = time.time() - cupid_start_time
                print(f"\nCUPID (TRAK) influence computed in {cupid_compute_time:.2f}s")
                print(f"Importance stats: min={cached_sample_importance.min():.6f}, "
                      f"max={cached_sample_importance.max():.6f}, "
                      f"mean={cached_sample_importance.mean():.6f}")
                print(f"{'='*60}\n")
            else:
                print("No successful samples found, using uniform sampling")
                cached_sample_importance = None
        elif use_cupid and not TRAK_AVAILABLE:
            print("Warning: CUPID enabled but TRAK not available. Using uniform sampling.")

        # Optimizing the policy and value network
        # After subsampling, we use the actual batch size (indexed 0 to N-1)
        actual_batch_size = b_obs.shape[0]
        b_inds = np.arange(actual_batch_size)
        clipfracs = []
        for epoch in trange(cfg.update_epochs, desc="Policy update"):
            early_stop = False

            np.random.shuffle(b_inds)
            
            # Process entire batch as one minibatch (since batch_size == minibatch_size)
            # CUPID: Weighted sampling based on importance scores
            if use_cupid and cached_sample_importance is not None:
                # Sample entire batch weighted by importance
                mb_inds = np.random.choice(
                    b_inds,
                    size=actual_batch_size,
                    p=cached_sample_importance,
                    replace=True  # Use replacement for weighted sampling
                )
            else:
                # Original: use all indices
                mb_inds = b_inds

            # Get the minibatch and place it on the device
            mb_obs = b_obs[mb_inds].to(device)
            mb_actions = b_actions[mb_inds].to(device)
            mb_logprobs = b_logprobs[mb_inds].to(device)
            mb_advantages = b_advantages[mb_inds].to(device)
            mb_returns = b_returns[mb_inds].to(device)
            mb_values = b_values[mb_inds].to(device)

            # Calculate the loss
            _, newlogprob, entropy, newvalue, action_mean = (
                residual_policy.get_action_and_value(mb_obs, mb_actions)
            )
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [
                    ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                ]

            if cfg.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            policy_loss = 0

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if cfg.clip_vloss:
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    newvalue - mb_values,
                    -cfg.clip_coef,
                    cfg.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            # Entropy loss
            entropy_loss = entropy.mean() * cfg.ent_coef

            ppo_loss = pg_loss - entropy_loss

            # Add the auxiliary regularization loss
            residual_l1_loss = torch.mean(torch.abs(action_mean))
            residual_l2_loss = torch.mean(torch.square(action_mean))

            # Normalize the losses so that each term has the same scale
            if iteration > cfg.n_iterations_train_only_value:

                # Scale the losses using the calculated scaling factors
                policy_loss += ppo_loss
                policy_loss += cfg.residual_l1 * residual_l1_loss
                policy_loss += cfg.residual_l2 * residual_l2_loss

            # Total loss
            loss: torch.Tensor = policy_loss + v_loss * cfg.vf_coef

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(
                residual_policy.parameters(), cfg.max_grad_norm
            )

            optimizer_actor.step()
            optimizer_critic.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                print(
                    f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > {cfg.target_kl:.4f}"
                )
                early_stop = True
                break

        if early_stop:
            break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        action_norms = torch.norm(b_actions[:, :3], dim=-1).cpu()

        training_cum_time += time.time() - iteration_start_time
        sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

        # Prepare logging dict
        log_dict = {
            "training/learning_rate_actor": optimizer_actor.param_groups[0]["lr"],
            "training/learning_rate_critic": optimizer_critic.param_groups[0]["lr"],
            "training/SPS": sps,
            "charts/rewards": rewards.sum().item(),
            "charts/success_rate": success_rate,
            "charts/success_timesteps_share": success_timesteps_share,
            "charts/mean_success_episode_length": mean_success_episode_length,
            "charts/max_success_episode_length": max_success_episode_length,
            "charts/action_norm_mean": action_norms.mean(),
            "charts/action_norm_std": action_norms.std(),
            "values/advantages": b_advantages.mean().item(),
            "values/returns": b_returns.mean().item(),
            "values/values": b_values.mean().item(),
            "values/mean_logstd": residual_policy.actor_logstd.mean().item(),
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/total_loss": loss.item(),
            "losses/entropy_loss": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "losses/residual_l1": residual_l1_loss.item(),
            "losses/residual_l2": residual_l2_loss.item(),
            "histograms/values": wandb.Histogram(values),
            "histograms/returns": wandb.Histogram(b_returns),
            "histograms/advantages": wandb.Histogram(b_advantages),
            "histograms/logprobs": wandb.Histogram(logprobs),
            "histograms/rewards": wandb.Histogram(rewards),
            "histograms/action_norms": wandb.Histogram(action_norms),
        }
        
        # CUPID: Add importance score metrics
        if use_cupid and cached_sample_importance is not None:
            log_dict.update({
                "cupid/importance_min": cached_sample_importance.min(),
                "cupid/importance_max": cached_sample_importance.max(),
                "cupid/importance_mean": cached_sample_importance.mean(),
                "cupid/importance_std": cached_sample_importance.std(),
                "cupid/top10_mean": np.sort(cached_sample_importance)[-int(0.1*len(cached_sample_importance)):].mean(),
                "cupid/bottom10_mean": np.sort(cached_sample_importance)[:int(0.1*len(cached_sample_importance))].mean(),
                "histograms/importance": wandb.Histogram(cached_sample_importance),
            })
        
        wandb.log(log_dict, step=global_step)

        # Step the learning rate scheduler
        lr_scheduler_actor.step()
        lr_scheduler_critic.step()

        # Checkpoint every cfg.checkpoint_interval steps
        if cfg.checkpoint_interval > 0 and iteration % cfg.checkpoint_interval == 0:
            model_path = str(model_save_dir / f"actor_chkpt_{iteration}.pt")
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                    "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                    "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                    "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "success_rate": success_rate,
                    "iteration": iteration,
                    "training_cum_time": training_cum_time,
                },
                model_path,
            )

            wandb.save(model_path)
            print(f"Model saved to {model_path}")

        # Print some stats at the end of the iteration
        print(
            f"Iteration {iteration}/{cfg.num_iterations}, global step {global_step}, SPS {sps}"
        )

    print(f"Training finished in {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    main()
