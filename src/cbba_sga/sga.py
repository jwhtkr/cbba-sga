"""Implement the SGA algorithm."""

import typing
from cbba_sga import agent, config, pool, task


class SgaConfig(config.ConfigBase):
    """Configuration for the SGA algorithm."""

    max_iterations: int = 100
    multiprocessing: bool = False
    chunk_size: int = 10


_DataT = typing.TypeVar("_DataT")
T = typing.TypeVar("T")
RT = typing.TypeVar("RT")


def map_(
    func: typing.Callable[[T], RT], iterable: typing.Iterable[T], _chunksize: int = 10
) -> typing.Iterable[RT]:
    """Wrap `map` to use chunksize (ignored) and be eager (match smart pool)."""
    return list(map(func, iterable))


def sga(
    agents: typing.Sequence[agent.Agent[_DataT, None]],
    tasks: typing.Sequence[task.Task[_DataT]],
    config_: SgaConfig,
) -> None:
    """Assign tasks to agents using the SGA algorithm."""
    open_tasks = [task_ for task_ in tasks if task_.available]
    for agent_ in agents:
        agent_.initialize_bids()
    _map = map_ if not config_.multiprocessing else pool.SmartPool().imap_unordered
    _map = typing.cast(
        typing.Callable[
            [
                typing.Callable[[agent.Agent[_DataT, None]], None],
                typing.Iterable[agent.Agent[_DataT, None]],
                int,
            ],
            typing.Iterable[None],
        ],
        _map,
    )

    for _i in range(config_.max_iterations):
        selection_result = select_best_assignment(agents, open_tasks)
        score, selected_agent_idx, selected_task_idx, assignment_data = selection_result
        selected_agent = agents[selected_agent_idx]
        selected_task = open_tasks[selected_task_idx]
        selected_task.assign_to(selected_agent.id, score, assignment_data)
        selected_agent.add_to_bundle(selected_task)

        def update_scores(agent_: agent.Agent[_DataT, None]) -> None:
            """Wrap the agent score update functionality."""
            # This `cell-var` is only used in the loop, and so is fine.
            # pylint: disable-next=cell-var-from-loop
            agent_.update_bids([selected_task])

        list(_map(
            update_scores,
            agents,
            config_.chunk_size,
        ))

        if not selected_task.available:
            open_tasks.remove(selected_task)
        if all(agent_.done for agent_ in agents) or not open_tasks:
            break


def select_best_assignment(
    agents: typing.Sequence[agent.Agent[_DataT, None]],
    tasks: typing.Sequence[task.Task[_DataT]],
) -> typing.Tuple[float, int, int, _DataT]:
    """Select the best assignment to add to the SGA assignments."""
    score = -float("inf")
    selected_agent_idx: int
    selected_task_idx: int
    assignment_data: _DataT
    for agent_idx, agent_ in enumerate(agents):
        if agent_.done:
            continue
        for task_idx, task_ in enumerate(tasks):
            bid = agent_.task_bid(task_)
            if bid is None:
                continue
            if bid.value > score:
                score = bid.value
                selected_agent_idx = agent_idx
                selected_task_idx = task_idx
                assignment_data = bid.data

    return score, selected_agent_idx, selected_task_idx, assignment_data
