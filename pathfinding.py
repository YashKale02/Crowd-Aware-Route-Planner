from heapq import heappush, heappop
import numpy as np

# Calculate heuristic distance (8-directional / Octile distance)
def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

# A* Pathfinding Algorithm
def astar(grid, start, goal, crowd_weight=25.0):
    CROWD_WEIGHT = crowd_weight

    # 8 possible movement directions (4 straight, 4 diagonal)
    neighbors = [(0,1), (1,0), (-1,0), (0,-1),
                 (1,1), (1,-1), (-1,1), (-1,-1)]

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heappush(oheap, (fscore[start], start))

    while oheap:
        _, current = heappop(oheap)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(current)
            return path[::-1]  # reverse to get start → goal

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            move_cost = np.sqrt(2) if i != 0 and j != 0 else 1

            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                new_g = gscore[current] + move_cost + (CROWD_WEIGHT * grid[neighbor])
                if neighbor in close_set and new_g >= gscore.get(neighbor, np.inf):
                    continue
                if new_g < gscore.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    gscore[neighbor] = new_g
                    fscore[neighbor] = new_g + heuristic(neighbor, goal)
                    heappush(oheap, (fscore[neighbor], neighbor))

    return None
