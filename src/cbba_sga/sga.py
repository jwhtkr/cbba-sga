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

    __mp_attributes = {
        "_agent",
        "_agent_cls",
        "_process",
        "_conn",
        "_recv_count",
        "initialize_bids",
        "task_bid",
        "add_to_bundle",
        "update_bids",
        "_flush",
    }

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
            daemon=True,
        )
        self._process.start()
        self._recv_count = 0

    def initialize_bids(self) -> None:
        self._conn.send((self._agent_cls.initialize_bids, (), {}))
        self._recv_count += 1

    def task_bid(
        self, task_: task.Task[_DataT]
    ) -> typing.Tuple[typing.Optional[agent.Bid[_DataT]], task.Task[_DataT]]:
        self._flush()
        self._conn.send((self._agent_cls.task_bid, (task_,), {}))
        out: typing.Tuple[typing.Optional[agent.Bid[_DataT]], task.Task[_DataT]]
        out = self._conn.recv()
        return out

    def add_to_bundle(self, task_: task.Task[_DataT], bid: agent.Bid[_DataT]) -> None:
        # We want to call the member agent's `add_to_bundle` here in order to
        # locally track the bundle and path.
        self._agent.add_to_bundle(task_, bid)
        # But, we still need to send the call to the process so that it is also
        # tracked in the "real" agent.
        self._conn.send((self._agent_cls.add_to_bundle, (task_, bid), {}))
        self._recv_count += 1

    def update_bids(self, update_tasks: typing.List[task.Task[_DataT]]) -> None:
        self._conn.send((self._agent_cls.update_bids, (update_tasks,), {}))
        self._recv_count += 1

    def _flush(self) -> None:
        """Flush the unneeded received messages from the connection.

        Use to ensure that synchronous functions (e.g., with return values) get
        the correct return value from the connection and/or are synchronized
        before moving on.
        """
        for _ in range(self._recv_count):
            self._conn.recv()
        self._recv_count = 0

    def __getattribute__(self, name: str) -> typing.Any:
        if name.startswith("__") or name in AgentProxy.__mp_attributes:
            return super().__getattribute__(name)
        return self._agent.__getattribute__(name)

    def __del__(self) -> None:
        logger.debug(f"Cleaning up agent {self._agent.id}'s process")
        self._conn.send(class_worker.STOP)
        self._conn.recv()
        self._process.join(0.1)
        if self._process.is_alive():
            self._process.terminate()
        self._conn.close()


def sga(
    agents: typing.Sequence[agent.Agent[_DataT, None]],
    tasks: typing.Sequence[task.Task[_DataT]],
    config_: SgaConfig,
) -> typing.List[agent.Agent[_DataT, None]]:
    """Assign tasks to agents using the SGA algorithm."""
    open_tasks = [task_ for task_ in tasks if task_.available]
    if config_.multiprocessing:
        agents = [AgentProxy(agent_) for agent_ in agents]

    for agent_ in agents:
        agent_.initialize_bids()

    for _i in range(config_.max_iterations):
        logger.debug("SGA iteration %s", _i)
        selection_result = select_best_assignment(agents, open_tasks)
        selected_bid, selected_agent_idx, selected_task_idx, assignment_data = (
            selection_result
        )

        selected_agent = agents[selected_agent_idx]
        selected_task = open_tasks[selected_task_idx]
        selected_task.assign_to(selected_agent.id, selected_bid.value, assignment_data)
        selected_agent.add_to_bundle(selected_task, selected_bid)

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

    return list(agents)


def select_best_assignment(
    agents: typing.Sequence[agent.Agent[_DataT, None]],
    tasks: typing.Sequence[task.Task[_DataT]],
) -> typing.Tuple[agent.Bid[_DataT], int, int, _DataT]:
    """Select the best assignment to add to the SGA assignments."""
    logger.debug("Selecting the best assignment.")
    selected_bid: typing.Optional[agent.Bid[_DataT]] = None
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
            if selected_bid is None or bid.value > selected_bid.value:
                logger.debug(
                    "The best assignment is updated to agent %s, task %s, and score %f",
                    agent_.id,
                    task_.id,
                    bid.value,
                )
                selected_bid = bid
                selected_agent_idx = agent_idx
                selected_task_idx = task_idx
                assignment_data = bid.data
    if selected_bid is None:
        raise RuntimeError(
            "No valid bid was found. This usually means a task or agent is "
            "not indicating it is done or unavailable properly."
        )

    return selected_bid, selected_agent_idx, selected_task_idx, assignment_data
