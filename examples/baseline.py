"""This example is essentially the one from the CBBA paper."""

import math
import typing
from cbba_sga import agent, sga, task


class PositionTask(task.Task[None]):
    """A Task that has a position."""

    def __init__(self, id_: str, position: typing.Tuple[float, float]) -> None:
        super().__init__(id_, max_agents=1)
        self.position = position
        self.value = 1.0


class PositionAgent(agent.Agent[None, None]):
    """An Agent that has a position."""

    def __init__(
        self,
        id_: str,
        tasks: typing.List[PositionTask],
        position: typing.Tuple[float, float],
        lambda_: float,
        max_tasks: int,
    ) -> None:
        super().__init__(id_, None, [], tasks)
        self.position = position
        self.lambda_ = lambda_
        self.max_tasks = max_tasks
        self.curr_path_score = 0.0
        self.bundle: typing.List[PositionTask]  # type: ignore[assignment]
        self.path: typing.List[PositionTask]  # type: ignore[assignment]
        self.winning_assignments: typing.List[PositionTask]
        self._set_args_kwargs(id_, tasks, position, lambda_, max_tasks)

    @property
    def done(self) -> bool:
        return len(self.bundle) >= self.max_tasks

    def initialize_bids(self) -> None:
        self.curr_bids = [None] * len(self.winning_assignments)
        for idx, task_ in enumerate(self.winning_assignments):
            self.curr_bids[idx] = agent.Bid(
                *self._calc_best_path_score(task_), data=None
            )

    def _update_bids_task_assignments(
        self, updated_tasks: typing.Iterable[task.Task[None]]
    ) -> None:
        for updated_task in updated_tasks:
            for task_idx, task_ in enumerate(self.winning_assignments):
                if updated_task.id == task_.id:
                    self.curr_bids[task_idx] = None

    def _update_bids_bundle_update(self) -> None:
        self.curr_path_score = self._calc_path_score(self.path)

        bid_task_zip = enumerate(zip(self.curr_bids, self.winning_assignments))
        for idx, (prev_bid, task_) in bid_task_zip:
            if prev_bid is None:
                continue
            score, path_idx = self._calc_best_path_score(task_)
            prev_bid.value = score
            prev_bid.path_idx = path_idx
            self.curr_bids[idx] = prev_bid

    def _calc_best_path_score(self, task_: PositionTask) -> typing.Tuple[float, int]:
        """Calculate the best score and index in the path for the task."""
        best_score = -float("inf")
        best_idx: int

        path_positions = [task_.position for task_ in self.path]
        for path_idx in range(len(path_positions) + 1):
            candidate_path = self.path[:path_idx] + [task_] + self.path[path_idx:]

            score = self._calc_path_score(candidate_path)

            if score > best_score:
                best_score = score
                best_idx = path_idx

        return best_score - self.curr_path_score, best_idx

    def _calc_path_score(self, path: typing.List[PositionTask]) -> float:
        """Calculate the score of a path."""
        positions = [self.position]
        positions.extend(task_.position for task_ in path)

        cum_distance = 0.0
        score = 0.0
        for distance, task_ in zip(calc_path_distances(positions), path):
            cum_distance += distance
            score += (self.lambda_**cum_distance) * task_.value
        return score


def calc_path_distances(
    path: typing.Sequence[typing.Tuple[float, ...]]
) -> typing.List[float]:
    """Calculate the distance the path travels."""
    distances = []
    for curr_pos, next_pos in zip(path, path[1:]):
        delta = math.sqrt(sum((one - two) ** 2 for one, two in zip(curr_pos, next_pos)))
        distances.append(delta)

    return distances


if __name__ == "__main__":
    import logging
    import os
    import random
    import matplotlib.pyplot as plt
    import time

    random.seed(42)

    logging.basicConfig(
        level=logging.DEBUG, filename=f"{os.curdir}baseline.log", filemode="w"
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    N = 20
    MP = False

    _tasks = [
        PositionTask(str(i), (random.uniform(-10, 10), random.uniform(-10, 10)))
        for i in range(5 * N)
    ]
    _agents = [
        PositionAgent(
            str(i), _tasks, (random.uniform(-10, 10), random.uniform(-10, 10)), 0.8, 6
        )
        for i in range(N)
    ]
    # _tasks = [
    #     PositionTask("1", (0, 1)),
    #     PositionTask("2", (0, 2)),
    #     PositionTask("3", (1, 2)),
    # ]
    # _agents = [PositionAgent("1", _tasks, (0, 0), 0.5, 3)]

    sga_start = time.perf_counter()
    sga.sga(_agents, _tasks, sga.SgaConfig(max_iterations=1000, multiprocessing=MP))
    sga_end = time.perf_counter()
    logging.info("SGA time: %f", sga_end - sga_start)
    print(f"SGA time: {sga_end - sga_start}")

    plt.figure()
    for _task in _tasks:
        plt.plot(*_task.position, "xk")

    _color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, _agent in enumerate(_agents):
        plt.plot(*_agent.position, "s", color=_color_cycle[i % 10])

        _path_positions = [_agent.position]
        _path_positions.extend(_task.position for _task in _agent.path)
        plt.plot(*zip(*_path_positions), color=_color_cycle[i % 10])

        _bundle_positions = [_agent.position]
        _bundle_positions.extend(_task.position for _task in _agent.bundle)
        # for _bundle_position
        plt.plot(*zip(*_bundle_positions), ":", color=_color_cycle[i % 10], linewidth=2)

    # plt.figure()
    # _n_agents = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
    # _sga_times = [0.008, 0.05, 0.2, 0.5, 1.1, 2.1, 3.7, 6.9, 14.7, 29.7]
    # # _coeffs = [0.0003, -0.0094, 0.1401, -0.5823]
    # _coeffs = [1.9e-6, 1.2e-5, 9.3e-4, -2.1e-2, 1.1e-1]
    # _approx = [
    #     sum(coeff * val ** (len(_coeffs) - i - 1)
    #     for i, coeff in enumerate(_coeffs))
    #     for val in _n_agents
    # ]
    # plt.plot(_n_agents, _sga_times)
    # plt.plot(
    #     _n_agents,
    #     _approx,
    # )
    plt.show()
