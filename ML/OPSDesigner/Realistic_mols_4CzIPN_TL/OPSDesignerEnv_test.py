# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:20:16 2025

@author: noton
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rdkit import Chem
from GCNcalculator import predict_properties
from T5generator import gen_smiles_from_fragments


class OPSDesignerEnv(gym.Env):
    """
    An RL environment in which the agent sequentially selects fragment SMILES from a predefined set,
    then generates a molecular SMILES using a T5-based model and evaluates its properties.

    - action:
        0 .. N-1 : select the corresponding fragment (duplicates allowed)
        N        : termination action

    - reward:
        Higher rewards are given when the predicted properties have a smaller RMSE relative to the target_ops,
        optionally combined with shaping (potential / delta / none).
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 smiles,
                 target_ops,
                 *,
                 min_k: int = 2,
                 max_k: int = 5,
                 max_steps: int = 6,
                 eps: float = 1e-8,
                 r_cap: int | None = 100,
                 shaping: str = "potential",
                 gamma: float = 0.99,
                 w_shaping: float = 1.0,
                 w_terminal: float = 1.0,
                 step_penalty: float = 0.0,
                 invalid_penalty: float = -10,
                 ):
        
        super().__init__()
        
        # List of available fragment SMILES and number of actions ("0"–"N-1":Fragment selection, "N":Termination)
        self.smiles = list(smiles)
        self.N = len(self.smiles)

        # Target OPS properties (EHOMO, ELUMO, ES1, fS1, ET1, DEST)
        self.target = np.asarray(target_ops, dtype=np.float32)
        assert self.target.shape == (6,), "target_ops must be shape (6,)"

        # Minimum and maximum number of fragments per episode
        self.min_k = int(min_k)
        self.max_k = int(max_k)
        
        # Maximum number of environment steps per episode
        self.max_steps = int(max_steps)
        
        # eps: small value to avoid division by zero
        # r_cap: upper bound on the terminal reward
        self.eps = float(eps)
        self.r_cap = None if r_cap is None else float(r_cap)

        # Reward shaping configuration
        self.shaping = str(shaping)          # shaping mode: {"potential", "delta", "none"}
        self.gamma = float(gamma)            # discount factor used in potential-based shaping
        self.w_shaping = float(w_shaping)    # weight for shaping rewards
        self.w_terminal = float(w_terminal)  # weight for terminal rewards

        # Penalties for taking a step and for invalid molecule generation
        self.step_penalty = float(step_penalty)
        self.invalid_penalty = float(invalid_penalty)

        # Definition of the observation space
        # obs = [ number_of_selected_fragments / max_k , normalized_RMSE ]
        obs_dim = 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Definition of the action space ("0"–"N-1":Fragment selection, "N":Termination)
        self.action_space = spaces.Discrete(self.N + 1)

        # Initialization of internal episode states
        self._reset_state()


    def _reset_state(self):
        """
        Helper function to initialize all internal states for a new episode.
        This method is called from reset().
        """
        # The sequence of selected fragment indices in the current episode
        self.selection_order: list[int] = []

        self.count = 0        # Number of fragments selected so far
        self.steps = 0        # Number of steps taken since the last reset() call
        self.done = False     # Episode termination flag
        self.last_rmse = 1.0  # Most recent RMSE value (initialized to 1.0 when evaluation is unavailable)

        # Information about the "best" generated molecule so far (SMILES, predicted properties, RMSE, etc.)
        self._last_best: dict | None = None
        

    def _current_frags(self):
        """
        Return the "SMILES" and "indices" of the fragments currently selected in the ongoing episode.
        """
        idx = list(self.selection_order)
        return [self.smiles[i] for i in idx], idx


    def _evaluate_selection(self):
        """
        Evaluate the current fragment selection (selection_order) by:
        1) Generating a molecular SMILES with the T5-based model
        2) Validating the generated SMILES using RDKit
        3) Predicting properties with the GCN model
        4) Computing the RMSE against the target properties
        and return the resulting RMSE.

        In cases of failure:
        - None : The number of selected fragments is less than min_k (evaluation is skipped)
        - inf  : Generation / validation / prediction failed
        """
        
        self._last_best = None  # Clear the previous "best molecule" information

        frags, _ = self._current_frags()
        # Do not evaluate if the number of fragments is below the "min_k"
        if len(frags) < self.min_k:
            return None
        print(frags)
        # --- 1) Molecular generation by Frag2OPST5
        s = gen_smiles_from_fragments(frags, max_new_tokens=128)
        if s is None or not isinstance(s, str) or not s.strip():
            # Store meta-information when generation itself fails (return "inf")
            self._last_best = {"all_invalid": True, "why": "generator_none"}
            return float("inf")
        print(s)
        # --- 2) Validation of the generated SMILES
        m = Chem.MolFromSmiles(s)
        if (m is None) or (m.GetNumAtoms() < 2) or (m.GetNumBonds() < 1):
            # Store meta-information when the generated SMILES is invalid (return "inf")
            self._last_best = {"all_invalid": True, "why": "invalid_smiles"}
            return float("inf")

        # --- 3) GCN-based prediction of molecular properties
        try:
            pred = np.asarray(predict_properties(s), dtype=np.float32)
        except Exception as e:
            # Treat any exception during prediction as invalid (return "inf")
            self._last_best = {"all_invalid": True, "why": f"predict_properties:{type(e).__name__}"}
            return float("inf")

        # Normalize the prediction shape to (6,)
        # Check for NaN or inf in the predictions
        if pred.ndim == 2 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim != 1 or pred.shape[0] != 6:
            self._last_best = {"all_invalid": True, "why": f"preds_shape:{pred.shape}"}
            return float("inf")
        if not np.isfinite(pred).all():
            self._last_best = {"all_invalid": True, "why": "preds_nonfinite"}
            return float("inf")

        # --- 4) Computing the RMSE against the target properties
        print(pred)
        diffs = pred - self.target
        rmse = float(np.sqrt(np.mean(diffs**2)))
        print(rmse)

        # Store the current result as the "best" molecule for this evaluation
        self._last_best = {"best_rmse": rmse,
                           "best_smiles": s,
                           "best_pred": pred.tolist()}

        return rmse


    def _phi(self, rmse):
        """
        Potential function φ(rmse) used for potential-based reward shaping.
        Smaller RMSE values result in larger potentials.
        """
        if rmse is None or not np.isfinite(rmse):
            return 0.0
        return 1.0 / (rmse + self.eps)

    def _obs(self):
        """
        Construct the observation vector returned to the agent based on the current internal state.
        i.e. constructing the input to the NN that drives the RL policy
        Observation format:
            [0]: Number of selected fragments / max_k  (scaled to 0–1)
            [1]: Normalized RMSE score: rmse_norm = 1 / (1 + max(rmse, 0))
        """
        rmse = self.last_rmse
        rmse_norm = 0.0 if not np.isfinite(rmse) else 1.0 / (1.0 + max(rmse, 0.0))
        return np.array([self.count / self.max_k, rmse_norm], dtype=np.float32)

    
    # === Gym API ===
    def reset(self, *, seed=None, options=None):
        """
        Gym API: Reset the environment.
        """
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        """
        Gym API: Perform one environment transition (one step).

        Parameters
        ----------
        action : int
            0..N-1 : select a fragment with the corresponding index
            N      : termination action (finish fragment selection)

        Returns
        -------
        obs : np.ndarray
            Observation of the next state.
        reward : float
            Reward obtained at this step.
        terminated : bool
            True if the episode ended naturally
            (e.g., termination action was chosen or max_k was reached).
        truncated : bool
            True if the episode was forcefully stopped due to a constraint
            (e.g., the maximum number of steps max_steps was reached).
        info : dict
            Additional information for debugging or logging purposes.
        """
        
        # Check that the action is within the valid action space
        assert self.action_space.contains(action)
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Step counter
        self.steps += 1
        terminated = False
        truncated = False

        # Store the previous RMSE and its potential value for shaping
        prev_rmse = self.last_rmse
        prev_phi = self._phi(prev_rmse)

        # Base reward = step penalty
        reward = self.step_penalty

        # ------ Apply the action ------
        # Termination action or Appending the selected fragment
        # Termination: "action == self.N" / "self.count >= self.max_k" / "self.steps >= self.max_steps"
        if action == self.N:
            terminated = True
        else:
            self.selection_order.append(action)
            self.count += 1
            if self.count >= self.max_k:
                terminated = True
                
        if (self.steps >= self.max_steps) and not terminated:
            truncated = True

        # ------ Evaluation (generation + property prediction + RMSE) ------
        # None: len(frags) < self.min_k / invalid_penalty: inf, NaN, ...
        rmse = self._evaluate_selection()
        eval_ok = False
        invalid_flag = False
        
        if rmse is None:
            pass
        elif np.isfinite(rmse):
            self.last_rmse = rmse
            eval_ok = True  # Valid evaluation result: update last_rmse
        else:
            invalid_flag = True
            self.last_rmse = float("inf")
            reward += self.invalid_penalty

        # ------ Reward shaping ------
        # Delta-based shaping: (previous RMSE - current RMSE) / Potential-based shaping: γ φ(s') - φ(s)
        shaping_reward = 0.0
        if self.shaping != "none":
            if self.shaping == "delta":
                if (np.isfinite(prev_rmse) and np.isfinite(self.last_rmse)):
                    shaping_reward = prev_rmse - self.last_rmse
            elif self.shaping == "potential":
                if eval_ok:
                    next_phi = self._phi(self.last_rmse)
                    shaping_reward = self.gamma * next_phi - prev_phi

        reward += self.w_shaping * shaping_reward

        # ------ Terminal reward ------
        # Applied when the episode ends (terminated or truncated) and last_rmse is finite
        if (terminated or truncated) and eval_ok:
            bonus = 1.0 / (self.last_rmse + self.eps)
            if self.r_cap is not None:
                bonus = min(bonus, self.r_cap)
            reward += self.w_terminal * bonus


        # Update the episode completion flag
        # Collect info such as selected fragments, RMSE, shaping term, ...
        self.done = terminated or truncated
        frags, idx = self._current_frags()
        info = {"terminated": terminated,
                "truncated": truncated,
                "selected_indices": idx,
                "selected_smiles": frags,
                "last_rmse": self.last_rmse,
                "shaping": shaping_reward,
                "prev_rmse": prev_rmse,
                "invalid": invalid_flag}

        # If _last_best contains "best molecule" information, merge it into info
        b = self._last_best
        if isinstance(b, dict) and ("best_rmse" in b):
            info.update({"best_rmse": b["best_rmse"],
                         "best_smiles": b["best_smiles"],
                         "best_pred": b["best_pred"]})
            
        print("prev_rmse =", prev_rmse)
        print("last_rmse =", self.last_rmse)
        print("shaping_reward =", shaping_reward)
        print("terminal_bonus =", (1.0 / (self.last_rmse + self.eps)) if ((terminated or truncated) and eval_ok) else 0.0)
        print("final_reward =", reward)

        return self._obs(), float(reward), terminated, truncated, info
