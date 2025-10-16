"""A* pathfinding algorithm for validating wall placements in Quoridor.

This module provides A* pathfinding to check if both players can still reach
their goal rows after a wall placement.
"""

from typing import List, Optional, Set, Tuple
import heapq


def astar_path_exists(
    start: Tuple[int, int],
    goal_row: int,
    board_size: int,
    h_walls: Set[Tuple[int, int]],
    v_walls: Set[Tuple[int, int]]
) -> bool:
    """Check if a path exists from start position to goal row using A*.

    Args:
        start: Starting (row, col) position
        goal_row: Target row to reach
        board_size: Size of the board (9 for standard Quoridor)
        h_walls: Set of horizontal wall positions (row, col)
        v_walls: Set of vertical wall positions (row, col)

    Returns:
        True if a path exists to the goal row
    """
    path, _ = astar_find_path(start, goal_row, board_size, h_walls, v_walls)
    return path is not None


def astar_find_path(
    start: Tuple[int, int],
    goal_row: int,
    board_size: int,
    h_walls: Set[Tuple[int, int]],
    v_walls: Set[Tuple[int, int]]
) -> Tuple[Optional[List[Tuple[int, int]]], int]:
    """Find shortest path from start position to goal row using A*.

    Args:
        start: Starting (row, col) position
        goal_row: Target row to reach
        board_size: Size of the board (9 for standard Quoridor)
        h_walls: Set of horizontal wall positions (row, col)
        v_walls: Set of vertical wall positions (row, col)

    Returns:
        Tuple of (path, distance) where:
        - path: List of positions from start to goal, or None if no path exists
        - distance: Length of the path (number of moves), or -1 if no path exists
    """
    # A* algorithm
    # Priority queue: (f_score, position)
    # f_score = g_score + heuristic
    open_set = [(manhattan_distance(start, goal_row), start)]
    came_from = {}

    # g_score: cost from start to this position
    g_score = {start: 0}

    # f_score: estimated total cost from start through this position to goal
    f_score = {start: manhattan_distance(start, goal_row)}

    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        # Skip if already visited
        if current in visited:
            continue

        visited.add(current)

        # Check if we reached the goal row
        if current[0] == goal_row:
            path = reconstruct_path(came_from, current)
            return path, len(path) - 1  # distance is path length - 1 (number of moves)

        # Explore neighbors
        for neighbor in get_neighbors(current, board_size, h_walls, v_walls):
            if neighbor in visited:
                continue

            # Cost to reach neighbor is always 1 (one move)
            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # This path to neighbor is better than any previous one
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, goal_row)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # No path found
    return None, -1


def get_neighbors(
    pos: Tuple[int, int],
    board_size: int,
    h_walls: Set[Tuple[int, int]],
    v_walls: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Get valid neighboring positions considering walls.

    Args:
        pos: Current (row, col) position
        board_size: Size of the board
        h_walls: Set of horizontal wall positions
        v_walls: Set of vertical wall positions

    Returns:
        List of valid neighboring positions
    """
    from ..game.moves import _get_adjacent_positions, _can_move_between

    neighbors = []
    adjacent = _get_adjacent_positions(pos, board_size)

    for adj_pos in adjacent:
        # Check if we can move to this adjacent position (no wall blocking)
        if _can_move_between(pos, adj_pos, h_walls, v_walls):
            neighbors.append(adj_pos)

    return neighbors


def manhattan_distance(pos: Tuple[int, int], goal_row: int) -> int:
    """Calculate Manhattan distance heuristic to goal row.

    Args:
        pos: Current (row, col) position
        goal_row: Target row

    Returns:
        Manhattan distance to goal row (just the row difference)
    """
    return abs(pos[0] - goal_row)


def reconstruct_path(
    came_from: dict,
    current: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Reconstruct path from start to goal (for debugging/visualization).

    Args:
        came_from: Dictionary mapping position to previous position
        current: Goal position

    Returns:
        List of positions from start to goal
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
