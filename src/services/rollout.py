"""Self-play episode rollout — drive both ε-greedy recurrent policies (T4.6).

``collect_episode`` rolls ONE env episode driving the cop team and the thief from
their :class:`~src.services.policy.RecurrentPolicy` instances (hidden states reset
per episode, eq 9 continuity), recording per-role step records that
:func:`src.services.episode_pad.pad_episode` widens into the centralized replay
buffer schema. Each step stores the pre-step LOCAL obs the agent acted on, the
next obs, the pre/post global state (train-time only), the chosen actions, the
per-agent rewards, and the next legal masks. The cop reward is the env's
``dec_pomdp`` shared signal (recovered by the learner's active-mean over cops).
"""

from __future__ import annotations

from random import Random

from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.services.policy import RecurrentPolicy


def _cop_keys(obs: dict) -> list[str]:
    """Return the sorted cop agent keys present in an obs dict (``cop_0``, ...)."""
    return sorted(key for key in obs if key.startswith("cop_"))


def collect_episode(  # noqa: PLR0913 — env + two policies + cfg + the 3 per-episode knobs are all distinct
    env: CopsRobbersEnv,
    cop_policy: RecurrentPolicy,
    thief_policy: RecurrentPolicy,
    cfg: dict,
    seed: int,
    epsilon: float,
    rng: Random,
) -> dict:
    """Roll one self-play episode; return per-role step records + capture flag.

    Args:
        env: A reset-able cop/thief env (``num_cops`` set by the curriculum stage).
        cop_policy: The cop-team acting policy (drives all cops in lockstep).
        thief_policy: The thief acting policy (single agent).
        cfg: Loaded config (unused dims flow through ``pad_episode``).
        seed: Spawn seed for the episode (reproducible).
        epsilon: ε for both policies' ε-greedy exploration this episode.
        rng: The shared exploration RNG (seeded).

    Returns:
        ``{"cop": [step,...], "thief": [step,...], "capture": bool}`` — per-role
        step records ready for :func:`src.services.episode_pad.pad_episode`.
    """
    obs, info = env.reset(seed=seed)
    cop_policy.reset()
    thief_policy.reset()
    cop_keys = _cop_keys(obs)
    cop_steps: list[dict] = []
    thief_steps: list[dict] = []
    terminated = False
    capture = False
    while not terminated:
        state = env.state()
        masks = info["action_mask"]
        cop_obs = [obs[key] for key in cop_keys]
        cop_acts = cop_policy.act(cop_obs, [list(masks[key]) for key in cop_keys], epsilon, rng, state)
        thief_act = thief_policy.act([obs["thief"]], [list(masks["thief"])], epsilon, rng, state)[0]
        joint = dict(zip(cop_keys, cop_acts, strict=True))
        joint["thief"] = thief_act
        nxt, reward, terminated, info = env.step(joint)
        nxt_state, nmask = env.state(), info["action_mask"]
        cop_steps.append(
            {
                "obs": cop_obs,
                "nxt": [nxt[key] for key in cop_keys],
                "state": state,
                "nxt_state": nxt_state,
                "acts": [int(a) for a in cop_acts],
                "rews": [reward[key] for key in cop_keys],
                "done": terminated,
                "nmask": [list(nmask[key]) for key in cop_keys],
            }
        )
        thief_steps.append(
            {
                "obs": [obs["thief"]],
                "nxt": [nxt["thief"]],
                "state": state,
                "nxt_state": nxt_state,
                "acts": [int(thief_act)],
                "rews": [reward["thief"]],
                "done": terminated,
                "nmask": [list(nmask["thief"])],
            }
        )
        obs = nxt
        capture = bool(info.get("capture"))
    return {"cop": cop_steps, "thief": thief_steps, "capture": capture}
