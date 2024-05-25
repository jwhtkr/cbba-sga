"""Define the common types used in this package."""
# pylint: disable=missing-function-docstring,missing-class-docstring

# import typing


# class Task(typing.Protocol):
#     id: str
#     max_agents: int
#     assigned_agent_ids: typing.List[str]
#     assigned_agent_scores: typing.List[float]
#     assignment_data: typing.List[typing.Any]

#     def assign_to(  # noqa: E704
#         self, agent_id: str, score: float, data: typing.Any
#     ) -> None: ...

#     @property
#     def available(self) -> bool: ...  # noqa: E704
