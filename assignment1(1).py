import heapq

class SearchProblem:
    """ This is an abstract class which outlines the structure of a search problem. """

    def getStartState(self):
        pass

    def isGoalState(self, state):
        pass

    def getSuccessors(self, state):
        pass

class NQueenProblem(SearchProblem):
    def __init__(self, N):
        self.N = N
        self.initial = tuple([-1] * N)
    
    def getStartState(self):
        return self.initial
    
    def isGoalState(self, state):
        return (state[-1] != -1) and (not any([self.conflicted(state, r, c) for c, r in enumerate(state)]))

    def conflict(self, row1, col1, row2, col2):
        return ((row1 == row2) or 
                (col1 == col2) or 
                (row1 - col1 == row2 - col2) or
                (row1 + col1 == row2 + col2))
    
    def conflicted(self, state, row, col):
        return any([self.conflict(state[c], c, row, col) for c in range(col)])

    def placeTo(self, state, row, col):
        newState = list(state)
        newState[col] = row
        return tuple(newState)

    def getSuccessors(self, state):
        if (state[-1] != -1):
            return []
        col = state.index(-1)
        return [(self.placeTo(state, r, col), 1) for r in range(self.N) if (not self.conflicted(state, r, col))]
    


class Node:
    """ A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. """
    
    def __init__(self, state, parent=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
    
    def path(self):
        path_back = []
        node = self
        while (node):
            path_back.append(node)
            node = node.parent
        return path_back[::-1]
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

class Stack:
    def __init__(self):
        self.list = []
    
    def push(self, value):
        self.list.append(value)
    
    def pop(self):
        if (self.isEmpty()):    return None
        return self.list.pop()
    
    def isEmpty(self):
        return len(self.list) == 0

class Queue:
    def __init__(self):
        self.list = []
    
    def push(self, value):
        self.list.append(value)
    
    def pop(self):
        if (self.isEmpty()):    return None
        return self.list.pop(0)
    
    def isEmpty(self):
        return len(self.list) == 0

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0
    
    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1
    
    def pop(self):
        _, _, item = heapq.heappop(self.heap)
        return item
    
    def isEmpty(self):
        return len(self.heap) == 0
    
    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if (item == i):
                if (p <= priority):
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class GraphSearch:
    def __init__(self, problem, fringe, heuristic=None):
        self.problem = problem
        self.fringe = fringe
        self.heuristic = heuristic
        self.explored = dict()
        self.frontier = dict()
        self.initState = self.problem.getStartState()
    
    def search(self):
        if (self.initState is None):    return []
        initNode = Node(self.initState)
        newArg = [initNode]
        if (isinstance(self.fringe, PriorityQueue)):
            newArg.append(0 if (self.heuristic is None) else self.heuristic(self.initState, self.problem))
        self.__addNodeToFringe(tuple(newArg))
        self.frontier[hash(self.initState)] = initNode
        while (not self.fringe.isEmpty()):
            node = self.fringe.pop()
            self.explored[hash(node.state)] = node
            self.frontier.pop(hash(node.state), None)
            if (self.problem.isGoalState(node.state)):  return node.state
            self.__expand(node)
        return None
    
    def __expand(self, node):
        for (successor, step_cost) in self.problem.getSuccessors(node.state):
            key = hash(successor)
            if (self.explored.get(key, 0) == 0):
                child_path_cost = node.path_cost + step_cost
                child = Node(state=successor, parent=node, path_cost=child_path_cost)
                if (self.frontier.get(key, 0) == 0):
                    newArg = [child]
                    if (isinstance(self.fringe, PriorityQueue)):
                        newArg.append(child_path_cost + (0 if (self.heuristic is None) else self.heuristic(child.state, self.problem)))
                    self.__addNodeToFringe(tuple(newArg))
                    self.frontier[key] = child
                else:
                    if (isinstance(self.fringe, Stack)): # Depth First Search
                        self.__addNodeToFringe((child, ))
                    elif (isinstance(self.fringe, Queue)): # Breadth First Search
                        pass
                    else:   # Astar
                        old_node = self.frontier[key]
                        if (child_path_cost < old_node.path_cost):
                            self.fringe.update(child, child_path_cost)
                            self.frontier.pop(hash(old_node.state), None)
                            self.frontier[key] = child

    def __addNodeToFringe(self, newArg):
        self.fringe.push(*newArg)

class SearchAlgorithm:
    @staticmethod
    def depthFirstSearch(problem):
        fringe = Stack()
        graph = GraphSearch(problem, fringe)
        return graph.search()

    @staticmethod
    def breadthFirstSearch(problem):
        fringe = Queue()
        graph = GraphSearch(problem, fringe)
        return graph.search()

    @staticmethod
    def aStarSearch(problem, heuristic=None):
        fringe = PriorityQueue()
        graph = GraphSearch(problem, fringe, heuristic)
        return graph.search()

# Test Depth First Search
problem = NQueenProblem(4)
result = SearchAlgorithm.depthFirstSearch(problem)
print("Result of Depth First Search: {}".format(result))

# Test Breadth First Search
result = SearchAlgorithm.breadthFirstSearch(problem)
print("Result of Breadth First Search: {}".format(result))

# A heuristic for NQueen Problem
def heuristic(state, problem):
    num_conflicts = 0
    for (c1, r1) in enumerate(state):
        for (c2, r2) in enumerate(state):
            if (r1, c1) != (r2, c2):
                num_conflicts += problem.conflict(r1, c1, r2, c2)
    return num_conflicts

# Test A Star Search
result = SearchAlgorithm.aStarSearch(problem)
print("Result of A Star Search: {}".format(result))
