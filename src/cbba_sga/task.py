"""Represent a task in the task allocation problem CBBA/SGA solves."""

import typing


Data = typing.TypeVar("Data")


class Task(typing.Generic[Data]):
    """Represent a task in CBBA/SGA."""

    def __init__(self, id_: str, max_agents: int) -> None:
        self.id = id_
        self.max_agents = max_agents
        self.assigned_agent_ids: typing.List[str] = []
        self.assigned_agent_scores: typing.List[float] = []
        self.assignment_data: typing.List[Data] = []

    def assign_to(self, agent_id: str, score: float, data: Data) -> None:
        """Assign this task to the given agent."""
        if not agent_id:
            raise ValueError("An empty agent id string was given.")
        if not self.available:
            raise RuntimeError(
                "An attempt was made to override an existing assignment."
            )
        self.assigned_agent_ids.append(agent_id)
        self.assigned_agent_scores.append(score)
        self.assignment_data.append(data)

    @property
    def available(self) -> bool:
        """Return True if this task is still available to be assigned."""
        return len(self.assigned_agent_ids) < self.max_agents

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Task):
            return NotImplemented
        return self.id == value.id

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"

    def __repr__(self) -> str:
        _str = self.__str__()
        to_insert = f",max_agents={self.max_agents}"
        repr_str = _str[:-1] + to_insert + _str[-1:]
        return repr_str
