"""Represent an agent in the task allocation problem CBBA/SGA solves."""

import copy
import dataclasses
import logging
import multiprocessing as mp
import typing

from cbba_sga import task


logger = logging.getLogger(__name__)


DataT = typing.TypeVar("DataT", covariant=True)
CommsT = typing.TypeVar("CommsT")


@dataclasses.dataclass
class Bid(typing.Generic[DataT]):
    """Encapsulate info about the score/bid an agent makes on a task."""

    value: float
    path_idx: int
    data: DataT


# Below: Disable too-many-instance-attributes (R0902)
class Agent(typing.Generic[DataT, CommsT]):  # pylint: disable=R0902
    """Represent an agent in CBBA/SGA.

    Attributes:
        id: The ID of the agent.
        in_queue: The incoming message queue for the agent. Set to None if only
            using SGA. Set to a multiprocessing Queue (from multiprocessing
            Manager) for using with CBBA.
        out_queues: The outgoing message queues for the agent. Set to empty for
            SGA and a list of multiprocessing Queues (from a multiprocessing
            Manager) for using with CBBA.
        winning_assignments: The list of tasks that track the agent's local (if
            CBBA) knowledge of the assignment of tasks to agents and the related
            score and other data.
        curr_bids: The agent's current bids for all the tasks.
        bundle: The list of tasks assigned to the agent in assignment order.
        path: The list of tasks assigned to the agent in performed order.
        timestamps: The list of most recent times this agent has received
            information originating from each other agent.
        bids_are_dirty: A boolean indicating that the bids of this agent need
            to be updated to reflect an update in other information of this
            agent.
    """

    def __init__(
        self,
        id_: str,
        in_queue: "typing.Optional[mp.Queue[CommsT]]",
        out_queues: typing.List["mp.Queue[CommsT]"],
        tasks: typing.Sequence[task.Task[DataT]],
    ) -> None:
        """Initialize.

        Args:
            id: The ID of the agent.
            in_queue: The incoming message queue for the agent. Set to None if
                only using SGA. Set to a multiprocessing Queue for using with
                CBBA.
            out_queues: The outgoing message queues for the agent. Set to an
                empty list for SGA and a list of multiprocessing Queues for
                using with CBBA.
            tasks: The list of tasks that of the problem.
        """
        self.id = id_
        self.in_queue = in_queue
        self.out_queues = out_queues
        self.winning_assignments = copy.deepcopy(tasks)
        self.curr_bids: typing.List[typing.Optional[Bid[DataT]]] = []
        self.bundle: typing.List[task.Task[DataT]] = []
        self.path: typing.List[task.Task[DataT]] = []
        self.timestamps: typing.List[float] = []
        self.bids_are_dirty: bool = False
        self.args: typing.Tuple[typing.Any, ...]
        self.kwargs: typing.Dict[str, typing.Any]
        self._set_args_kwargs(id_, in_queue, out_queues, tasks)
        self._task_id_to_idx = {task_.id: idx for idx, task_ in enumerate(tasks)}

    @property
    def done(self) -> bool:
        """Return true if the agent is done assigning tasks."""
        raise NotImplementedError

    def add_to_bundle(self, task_: task.Task[DataT], bid: Bid[DataT]) -> None:
        """Add a task (with its data) to the agent's bundle."""
        if self.done:
            raise RuntimeError(
                "An attempt was made to add a task to an agent that is done."
            )
        logger.debug("Adding task %s to agent %s's bundle and path.", task_.id, self.id)
        my_task = self.get_task(task_.id)
        my_task.assign_to(self.id, bid.value, bid.data)
        self.path.insert(bid.path_idx, my_task)
        self.bundle.append(my_task)

        self.bids_are_dirty = True

    def initialize_bids(self) -> None:
        """Initialize the bids for all of the tasks."""
        raise NotImplementedError

    def task_bid(
        self, task_: task.Task[DataT]
    ) -> typing.Tuple[typing.Optional[Bid[DataT]], task.Task[DataT]]:
        """Retrieve the bid for a task."""
        try:
            idx = self._task_id_to_idx[task_.id]
        except ValueError as err:
            raise ValueError(
                f"Task {task_.id} was not found for agent {self.id}"
            ) from err
        return self.curr_bids[idx], self.winning_assignments[idx]

    def get_task(self, task_id: str) -> task.Task[DataT]:
        """Get the task from the given ID."""
        try:
            idx = self._task_id_to_idx[task_id]
        except ValueError as err:
            raise ValueError(
                f"Task {task_id} was not found for agent {self.id}"
            ) from err
        return self.winning_assignments[idx]

    def update_bids(self, update_tasks: typing.List[task.Task[DataT]]) -> None:
        """Update the bids of this agent based on the updated tasks."""
        logger.debug("Updating bids of agent %s.", self.id)
        self._update_bids_task_assignments(update_tasks)
        if self.bids_are_dirty:
            self._update_bids_bundle_update()
            self.bids_are_dirty = False

    def _update_bids_bundle_update(self) -> None:
        """Update the bids based on a new bundle."""
        raise NotImplementedError

    def _update_bids_task_assignments(
        self, updated_tasks: typing.Iterable[task.Task[DataT]]
    ) -> None:
        """Update the bids based on task assignments being updated."""
        raise NotImplementedError

    def _set_args_kwargs(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """Save the args and kwargs used to instantiate the class."""
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"

    def __repr__(self) -> str:
        _str = self.__str__()
        to_insert = f",tasks={self.winning_assignments}"
        repr_str = _str[:-1] + to_insert + _str[-1:]
        return repr_str
