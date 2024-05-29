"""Implement the SGA algorithm."""

import itertools
import logging
import typing

from cbba_sga import agent, config, pool, task


logger = logging.getLogger(__name__)


class SgaConfig(config.ConfigBase):
    """Configuration for the SGA algorithm."""

    max_iterations: int = 100
    multiprocessing: bool = False
    chunk_size: int = 10


_DataT = typing.TypeVar("_DataT")
T = typing.TypeVar("T")
RT = typing.TypeVar("RT")


def sga(
    agents: typing.Sequence[agent.Agent[_DataT, None]],
    tasks: typing.Sequence[task.Task[_DataT]],
    config_: SgaConfig,
) -> None:
    """Assign tasks to agents using the SGA algorithm."""
    open_tasks = [task_ for task_ in tasks if task_.available]
    mp_pool = None if not config_.multiprocessing else pool.SmartPool()

    if mp_pool is not None:
        proxy_agents = mp_pool.map(initialize_bids, agents)
        _update_agents_from_proxies(agents, proxy_agents)
    else:
        list(map(initialize_bids, agents))

    for _i in range(config_.max_iterations):
        logger.debug("SGA iteration %s", _i)
        selection_result = select_best_assignment(agents, open_tasks)
        score, selected_agent_idx, selected_task_idx, assignment_data = selection_result
        selected_agent = agents[selected_agent_idx]
        selected_task = open_tasks[selected_task_idx]
        selected_task.assign_to(selected_agent.id, score, assignment_data)
        selected_agent.add_to_bundle(selected_task)

        update_scores_zip = zip(agents, itertools.repeat(selected_task))
        if mp_pool is not None:
            proxy_agents = mp_pool.imap(
                update_scores,
                update_scores_zip,
                config_.chunk_size,
            )
            _update_agents_from_proxies(agents, proxy_agents)
        else:
            list(map(update_scores, update_scores_zip))

        if not selected_task.available:
            logger.debug("Removing task %s from the open tasks.", selected_task.id)
            open_tasks.remove(selected_task)
        if all(agent_.done for agent_ in agents):
            logger.debug("All agents are done. Stopping SGA.")
            break
        if not open_tasks:
            logger.debug("No more available tasks. Stopping SGA.")
            break

    if mp_pool is not None:
        logger.debug("Stopping the multiprocessing pool.")
        mp_pool.stop()


def select_best_assignment(
    agents: typing.Sequence[agent.Agent[_DataT, None]],
    tasks: typing.Sequence[task.Task[_DataT]],
) -> typing.Tuple[float, int, int, _DataT]:
    """Select the best assignment to add to the SGA assignments."""
    logger.debug("Selecting the best assignment.")
    score = -float("inf")
    selected_agent_idx: int
    selected_task_idx: int
    assignment_data: _DataT
    for agent_idx, agent_ in enumerate(agents):
        if agent_.done:
            continue
        for task_idx, task_ in enumerate(tasks):
            bid, _ = agent_.task_bid(task_)
            if bid is None:
                continue
            if bid.value > score:
                logger.debug(
                    "The best assignment is updated to agent %s, task %s, and score %f",
                    agent_.id,
                    task_.id,
                    bid.value,
                )
                score = bid.value
                selected_agent_idx = agent_idx
                selected_task_idx = task_idx
                assignment_data = bid.data

    return score, selected_agent_idx, selected_task_idx, assignment_data


def update_scores(
    args: typing.Tuple[agent.Agent[_DataT, None], task.Task[_DataT]]
) -> agent.Agent[_DataT, None]:
    """Wrap the agent score update functionality."""
    agent_, selected_task = args
    agent_.update_bids([selected_task])
    return agent_


def initialize_bids(agent_: agent.Agent[_DataT, None]) -> agent.Agent[_DataT, None]:
    """Initialize the agents bids (wraps agent function for pickling)."""
    agent_.initialize_bids()
    return agent_


def _update_agents_from_proxies(
    agents: typing.Sequence[agent.Agent[_DataT, None]],
    proxies: typing.Iterable[agent.Agent[_DataT, None]],
) -> None:
    """Update the agents to match the proxies from multiprocessing."""
    logger.debug("Copying new bids from multiprocessing proxy agents.")
    for agent_, proxy in zip(agents, proxies):
        if agent_.id != proxy.id:
            raise (
                ValueError(
                    "The order of the agents does not match the order of the proxies!"
                )
            )
        agent_.curr_bids = proxy.curr_bids
