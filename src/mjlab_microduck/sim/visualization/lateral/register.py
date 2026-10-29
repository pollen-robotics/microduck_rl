"""Task-registry integration kept with the optional lateral extension."""

TASK_ID = "Mjlab-Lateral-Flat-Backlash-MicroDuck"


def register_lateral_task(runner_cls) -> None:
    """Keep the existing train/list-envs task name without editing base recipes."""
    from mjlab.tasks.registry import register_mjlab_task

    # The task package calls this hook after loading all upstream recipes.
    # Bypass the public facade's bootstrap to avoid importing that package twice.
    from .env_cfg import MicroduckLateralRlCfg, make_lateral_cfg

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_lateral_cfg(),
        play_env_cfg=make_lateral_cfg(play=True),
        rl_cfg=MicroduckLateralRlCfg,
        runner_cls=runner_cls,
    )
