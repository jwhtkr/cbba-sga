"""Represent an agent in the task allocation problem CBBA/SGA solves."""

import copy
import dataclasses
import multiprocessing as mp
import typing

from cbba_sga import task


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
    """Represent an agent in CBBA/SGA."""

    def __init__(
        self,
        id_: str,
        in_queue: "mp.Queue[CommsT]",
        out_queues: typing.List["mp.Queue[CommsT]"],
        tasks: typing.Sequence[task.Task[DataT]],
    ) -> None:
        self.id = id_
        self.in_queue = in_queue
        self.out_queues = out_queues
        self.winning_assignments = copy.deepcopy(tasks)
        self.curr_bids: typing.List[typing.Optional[Bid[DataT]]] = []
        self.bundle: typing.List[task.Task[DataT]] = []
        self.path: typing.List[task.Task[DataT]] = []
        self.timestamps: typing.List[float] = []
        self.scores_are_dirty: bool = False

    @property
    def done(self) -> bool:
        """Return true if the agent is done assigning tasks."""
        raise NotImplementedError

    def add_to_bundle(self, task_: task.Task[DataT]) -> None:
        """Add a task (with its data) to the agent's bundle."""
        if self.done:
            raise RuntimeError(
                "An attempt was made to add a task to an agent that is done."
            )

        bid = self.task_bid(task_)
        if bid is None:
            raise ValueError(
                f"Task {task_.id} is not a valid task to add to agent "
                f"{self.id}'s bundle at this time."
            )
        self.path.insert(bid.path_idx, task_)
        self.bundle.append(task_)

        self.scores_are_dirty = True

    def initialize_bids(self) -> None:
        """Initialize the bids for all of the tasks."""
        raise NotImplementedError

    def update_bids(self, update_tasks: typing.List[task.Task[DataT]]) -> None:
        """Update the bids of this agent based on the updated tasks."""
        self._update_bids_task_assignments(update_tasks)
        if self.scores_are_dirty:
            self._update_bids_bundle_update()
            self.scores_are_dirty = False

    def _update_bids_bundle_update(self) -> None:
        """Update the bids based on a new bundle."""
        raise NotImplementedError

    def _update_bids_task_assignments(
        self, updated_tasks: typing.Iterable[task.Task[DataT]]
    ) -> None:
        """Update the bids based on task assignments being updated."""
        raise NotImplementedError

    def task_bid(self, task_: task.Task[DataT]) -> typing.Optional[Bid[DataT]]:
        """Retrieve the bid for a task."""
        for _bid, _task in zip(self.curr_bids, self.winning_assignments):
            if _task.id == task_.id:
                return _bid
        raise RuntimeError(f"Task {task_.id} could not be found for agent {self.id}")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"

    def __repr__(self) -> str:
        _str = self.__str__()
        to_insert = f",tasks={self.winning_assignments}"
        repr_str = _str[:-1] + to_insert + _str[-1:]
        return repr_str
