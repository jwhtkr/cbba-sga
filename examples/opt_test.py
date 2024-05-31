"""Test using optimization for per-agent scoring and multiprocessing speed."""

import typing

import gurobipy as gp
import numpy as np
import numpy.typing as npt

from cbba_sga import agent, task


def distance_matrix(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate the distance matrix."""
    n_points = points.shape[0]
    dist_mat = np.zeros((n_points, n_points))
    for i, point in enumerate(points):
        for j, other_point in enumerate(points):
            dist = np.linalg.norm(point - other_point)
            dist_mat[i, j] = dist
    return dist_mat


def build_model(
    start_pos: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    utilities: npt.NDArray[np.float64],
    max_points: int,
) -> gp.Model:
    """Build a Gurobi model of the single-agent problem."""
    if points.shape[0] != utilities.shape[0]:
        raise ValueError("Number of points and utility values must match")
    n_points = points.shape[0]
    dist_mat = distance_matrix(points)
    dist_mat += np.diag(
        np.ones_like(utilities) * (dist_mat.max() + utilities.max()) * 10
    )
    # print(f"Point distances:\n{dist_mat}")
    start_dists: npt.NDArray[np.float64] = np.linalg.norm(
        start_pos[np.newaxis, :] - points, axis=1
    )  # type:ignore[assignment]
    # print(f"Start distances: {start_dists}")

    model = gp.Model()
    model.params.LogToConsole = 0

    # Create model variables (with objective coefficients)
    point_edges = model.addMVar(
        dist_mat.shape, vtype=gp.GRB.BINARY, obj=-dist_mat, name="point_edge"
    )
    point_nodes = model.addMVar(
        (n_points,),
        obj=utilities,
        vtype=gp.GRB.CONTINUOUS,
        lb=0.0,
        ub=1.0,
        name="point",
    )
    point_idx = model.addMVar(
        (n_points,),
        obj=0.0,
        vtype=gp.GRB.CONTINUOUS,
        lb=0.0,
        ub=max_points,
        name="point_idx",
    )
    start_edges = model.addMVar(
        (n_points,), obj=-start_dists, vtype=gp.GRB.BINARY, name="start_edge"
    )
    end_edges = model.addMVar(
        (n_points,), obj=0.0, vtype=gp.GRB.BINARY, name="end_edge"
    )

    # Set model sense
    model.setAttr("ModelSense", gp.GRB.MAXIMIZE)

    # Add constraints
    point_edges_in = point_edges.sum(axis=0)
    point_edges_out = point_edges.sum(axis=1)
    model.addConstr(point_edges_in + start_edges <= 1, name="one_in")
    model.addConstr(point_edges_out + end_edges <= 1, name="one_out")
    model.addConstr(
        point_edges_in + start_edges == point_edges_out + end_edges, name="flow_balance"
    )
    model.addConstr(start_edges.sum() == 1, name="one_start")
    model.addConstr(end_edges.sum() == 1, name="one_end")
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            model.addConstr(
                point_idx[i] + 1
                <= point_idx[j] + (max_points + 1) * (1 - point_edges[i, j]),
                name=f"mtz[{i},{j}]",
            )
    model.addConstr(point_idx >= start_edges, name="mtz_start")
    model.addConstr(
        point_idx <= max_points * (point_edges_in + start_edges), name="mtz_zeros"
    )
    model.addConstr(point_nodes <= point_idx, name="nodes_zero")

    return model


class PositionTask(task.Task[None]):
    """A Task that has a position."""

    def __init__(
        self, id_: str, position: typing.Tuple[float, float], value: float
    ) -> None:
        super().__init__(id_, max_agents=1)
        self.position = position
        self.value = value


class PositionAgent(agent.Agent[None, None]):
    """An Agent that has a position."""

    def __init__(
        self,
        id_: str,
        tasks: typing.List[PositionTask],
        position: typing.Tuple[float, float],
        max_tasks: int,
    ) -> None:
        super().__init__(id_, None, [], tasks)
        self.position = position
        self.max_tasks = max_tasks
        self.curr_path_score = 0.0
        self.bundle: typing.List[PositionTask]  # type: ignore[assignment]
        self.path: typing.List[PositionTask]  # type: ignore[assignment]
        self.winning_assignments: typing.List[PositionTask]
        self._model = build_model(
            np.array(self.position),
            np.array([task_.position for task_ in tasks]),
            np.array([task_.value for task_ in tasks]),
            self.max_tasks,
        )
        self._model.params.LogFile = f"agent_{id_}.log"

    @property
    def done(self) -> bool:
        return len(self.bundle) >= self.max_tasks

    def initialize_bids(self) -> None:
        self.curr_bids = [None] * len(self.winning_assignments)
        self._calculate_bids()

    def add_to_bundle(self, task_: task.Task[None]) -> None:
        # Call base class to add task_ to bundle and path appropriately
        super().add_to_bundle(task_)

        # Add constraints for the newly added task_ to respect the path order
        bid_, _ = self.task_bid(task_)
        if bid_ is None:
            raise RuntimeError("This shouldn't be possible")
        task_before = self.path[bid_.path_idx - 1] if bid_.path_idx > 0 else None
        task_after = (
            self.path[bid_.path_idx + 1] if bid_.path_idx < len(self.path) - 1 else None
        )
        if task_before is not None:
            self._add_ordering_constraint(task_before, task_)
        if task_after is not None:
            self._add_ordering_constraint(task_, task_after)

    def _update_bids_task_assignments(
        self, updated_tasks: typing.Iterable[task.Task[None]]
    ) -> None:
        for task_idx, task_ in enumerate(self.winning_assignments):
            for updated_task in updated_tasks:
                if updated_task.id == task_.id:
                    # Copy data into owned task
                    task_.assigned_agent_ids = updated_task.assigned_agent_ids
                    task_.assigned_agent_scores = updated_task.assigned_agent_scores
                    task_.assignment_data = updated_task.assignment_data

                    # Add appropriate constraints
                    if not task_.available and self.id not in task_.assigned_agent_ids:
                        task_var = _get_task_var(self._model, task_idx)
                        self._model.addConstr(task_var == 0)
                        # Indicate the bids are no longer valid
                        self.bids_are_dirty = True

    def _update_bids_bundle_update(self) -> None:
        # TODO: update here
        self._calculate_bids()

    def _calculate_bids(self) -> None:
        """Calculate the bids for all tasks."""
        self._model.optimize()
        opt_val = self._model.ObjVal
        for idx, _task in enumerate(self.winning_assignments):
            task_var = _get_task_var(self._model, idx)
            if task_var.X < 0.5:
                continue

            loo_model = _leave_one_out_model(self._model, idx)
            loo_model.optimize()
            loo_opt_val = loo_model.ObjVal

            task_path_idx = _get_task_path_idx(self._model, idx)
            self.curr_bids[idx] = agent.Bid(opt_val - loo_opt_val, task_path_idx, None)

    def _add_ordering_constraint(
        self, earlier_task: task.Task[None], later_task: task.Task[None]
    ) -> None:
        """Constrain the earlier task to occur before the later task."""
        earlier_idx = -1
        later_idx = -1
        for idx, task_ in enumerate(self.winning_assignments):
            if task_.id == earlier_task.id:
                earlier_idx = idx
            if task_.id == later_task.id:
                later_idx = idx
        earlier_path_idx_var = _get_task_path_idx_var(self._model, earlier_idx)
        later_path_idx_var = _get_task_path_idx_var(self._model, later_idx)

        self._model.addConstr(earlier_path_idx_var <= later_path_idx_var)


def _leave_one_out_model(model: gp.Model, idx: int) -> gp.Model:
    """Create a leave one out model derived from the given model."""
    loo_model = model.copy()
    loo_var = _get_task_var(loo_model, idx)
    loo_model.addConstr(loo_var == 0)
    return loo_model


def _get_task_var(model: gp.Model, idx: int) -> gp.Var:
    """Retrieve the indicator variable for the task at the given index."""
    var = model.getVarByName(f"point[{idx}]")
    if var is None:
        raise RuntimeError("The variable name wasn't correct.")
    return var


def _get_task_path_idx_var(model: gp.Model, idx: int) -> gp.Var:
    """Retrieve the path index variable for the task at the given index."""
    var = model.getVarByName(f"point_idx[{idx}]")
    if var is None:
        raise RuntimeError("The variable name wasn't correct.")
    return var


def _get_task_path_idx(model: gp.Model, idx: int) -> int:
    """Retrieve the path index variable (and adjust to zero indexing)."""
    var = _get_task_path_idx_var(model, idx)
    return round(var.X) - 1


if __name__ == "__main__":
    import logging
    import os
    import time

    import matplotlib.pyplot as plt

    from cbba_sga import sga

    np.random.seed(42)

    _n_tasks = 8
    _n_agents = 4
    MP = False

    logging.basicConfig(
        level=logging.DEBUG, filename=f"{os.curdir}/opt_test.log", filemode="w"
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _tasks = [
        PositionTask(str(i), tuple(np.random.uniform(-10, 10, (2,)).tolist()), 30.0)
        for i in range(_n_tasks)
    ]
    _agents = [
        PositionAgent(
            str(i), _tasks, tuple(np.random.uniform(-10, 10, (2,)).tolist()), 3
        )
        for i in range(_n_agents)
    ]

    sga_start = time.perf_counter()
    sga.sga(
        _agents,
        _tasks,
        sga.SgaConfig(max_iterations=1000, multiprocessing=MP, chunk_size=1),
    )
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

    plt.show()
