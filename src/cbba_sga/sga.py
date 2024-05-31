"""Implement the SGA algorithm."""

import logging
import multiprocessing as mp
import typing

from cbba_sga import agent, class_worker, config, task


logger = logging.getLogger(__name__)


class SgaConfig(config.ConfigBase):
    """Configuration for the SGA algorithm."""

    max_iterations: int = 100
    multiprocessing: bool = False
    chunk_size: int = 10


_DataT = typing.TypeVar("_DataT")
T = typing.TypeVar("T")
RT = typing.TypeVar("RT")


class AgentProxy(agent.Agent[_DataT, None]):
    """A proxy that dispatches all calls to an agent in a process."""

    def __init__(self, agent_: agent.Agent[_DataT, None]) -> None:
        """Initialize.

        Args:
            agent_: The agent for which this class will be proxy.
        """
        self._agent = agent_
        self._agent_cls = type(agent_)
        parent_conn, child_conn = mp.Pipe()
        self._conn = parent_conn
        self._process = mp.Process(
            target=class_worker.class_worker,
            args=(child_conn, self._agent_cls, self._agent.args, self._agent.kwargs),
        )
        self._process.start()
        self._recv_stack: typing.List[None] = []

    @property
    def done(self) -> bool:
        self._flush()
        self._conn.send((self._agent_cls.done.__get__, (), {}))
        return typing.cast(bool, self._conn.recv())

    def initialize_bids(self) -> None:
        self._conn.send((self._agent_cls.initialize_bids, (), {}))
        self._recv_stack.append(None)

    def task_bid(
        self, task_: task.Task[_DataT]
    ) -> typing.Tuple[typing.Optional[agent.Bid[_DataT]], task.Task[_DataT]]:
        self._flush()
        self._conn.send((self._agent_cls.task_bid, (task_,), {}))
        out: typing.Tuple[typing.Optional[agent.Bid[_DataT]], task.Task[_DataT]]
        out = self._conn.recv()
        return out

    def add_to_bundle(self, task_: task.Task[_DataT]) -> None:
        # TODO: implement
        pass

    def _update_bids_task_assignments(
        self, updated_tasks: typing.Iterable[task.Task[_DataT]]
    ) -> None:
        # TODO: implement
        pass

    def _update_bids_bundle_update(self) -> None:
        # TODO: implement
        pass

    def _flush(self) -> None:
        """Flush the unneeded received messages from the connection.

        Use to ensure that synchronous functions (e.g., with return values) get
        the correct return value from the connection and/or are synchronized
        before moving on.
        """
        for _ in self._recv_stack:
            self._conn.recv()
        self._recv_stack.clear()


def sga(
    agents: typing.Sequence[agent.Agent[_DataT, None]],
    tasks: typing.Sequence[task.Task[_DataT]],
    config_: SgaConfig,
) -> None:
    """Assign tasks to agents using the SGA algorithm."""
    open_tasks = [task_ for task_ in tasks if task_.available]
    if config_.multiprocessing:
        agents = [AgentProxy(agent_) for agent_ in agents]

    for agent_ in agents:
        agent_.initialize_bids()

    for _i in range(config_.max_iterations):
        logger.debug("SGA iteration %s", _i)
        # TODO: Have select_best_assignment return the winning bid
        # This will be in place of `score`. Then, the Agent class will receive
        # the bid as an argument to the `add_to_bundle` method. This will reduce
        # the number of calls to `task_bid`, and will enable tracking the bundle
        # and path locally as well as in the agent's process.
        selection_result = select_best_assignment(agents, open_tasks)
        score, selected_agent_idx, selected_task_idx, assignment_data = selection_result

        selected_agent = agents[selected_agent_idx]
        selected_task = open_tasks[selected_task_idx]
        selected_task.assign_to(selected_agent.id, score, assignment_data)
        selected_agent.add_to_bundle(selected_task)

        for agent_ in agents:
            agent_.update_bids([selected_task])

        if not selected_task.available:
            logger.debug("Removing task %s from the open tasks.", selected_task.id)
            open_tasks.remove(selected_task)
        if all(agent_.done for agent_ in agents):
            logger.debug("All agents are done. Stopping SGA.")
            break
        if not open_tasks:
            logger.debug("No more available tasks. Stopping SGA.")
            break


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
