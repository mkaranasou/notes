Bitwise operations:
---
https://www.codeproject.com/Articles/544990/Understand-how-bitwise-operators-work-Csharp-and-V#dectobinandbintodec
Bitwise operators are used for numbers.
Bitwise operators perform an action on the bits of a number.

From decimal to binary:
```
Division:	783/2	391/2	195/2	97/2	48/2	24/2	12/2	6/2	3/2	1/2
Quotient:	391	195	97	48	24	12	6	3	1	0
Remainder:	1	1	1	1	0	0	0	0	1	1
```

Read the sequence of remainders (1111000011) from **right to left**, then you read 1100001111.
So, 1100001111v2 is 783v10.

To convert a negative decimal number to binary (-783 for example):
-   Take the binary form of 783: 0000001100001111.
-   Invert it: 1111110011110000
-   Add up 1111110011110000 with 1.

So, -783v10 is 111111001111000v12

How you can be sure this number is negative?

It depends on the data type.
If the data type is an Int16, then if the first bit is a 0, then the number is positive.
If the first bit is a 1, the number is negative.
So, 1111110011110000 (Int16) is -783, but for an unsigned number,
UInt16 for example, the first number DOESN'T tell whether the number is negative or not.
For an UInt16 for example, we can be sure it's positive because it's unsigned. So, 1111110011110000 as an UInt16 is 64752.

- OR (Inclusive OR) ( | )
- AND ( & )
- XOR (Exclusive OR) ( ^ )
- NOT ( ~ )
- Left Shift ( << )
- Right Shift ( >> )
- Circular Shift
  - Circular Left Shift (no operator in C# and VB.NET)
  - Circular Right Shift (no operator in C# and VB.NET)

Algorithms
---
`source: Grokking Algorithms`
### Big O notation:

Big O notation lets you compare the number of operations. It
tells you how fast the algorithm grows. Algorithm speed isn’t measured in seconds,
but in growth of the number of operations, which means, how quickly the run time
of an algorithm increases as the size of the input increases
Big O establishes worst case run time.

- O(log n), also known as log time. Example: Binary search.
- O(n), also known as linear time. Example: Simple search.
- O(n * log n). Example: A fast sorting algorithm, like quicksort
- O(n^2). Example: A slow sorting algorithm, like selection sort
- O(n!). Example: A really slow algorithm, like the traveling salesperson

### Logarithms:
---
“How many 10s do we multiply
together to get 100?” 
The answer is 2: 10 × 10. So log10 100 = 2. 
Logs are the flip of exponentials.

### Binary Search:
---
If you are searching in a list with n length, start from n/2 and work your way
to your solution - the list must be ordered.
Complexity is O(log(n))

Traveling Salesman:
- Look at every possible order in which he could travel to the cities
- Adds up the total distance
- Pick the path with the lowest distance
So, there are 120 permutations with 5 cities, so it will take 120 operations to solve the problem for 5 cities.
For n items, it will take n! (n factorial) operations to compute the result.

### Arrays vs Linked Lists:
---
Arrays: Read O(1), Write O(n), Delete O(n)

Lists: Read O(n), Write O(1), Delete O(1)


### Selection sort:
---
O(n2)

```python
def find_smallest(arr):
  smallest = arr[0]   # will store the smallest value - instantiate with first element
  smallest_index = 0  # will the index of the smallest value - initial is 0
  for i in range(1, len(arr)):
    if arr[i] < smallest:
      smallest = arr[i]
      smallest_index = i
  return smallest_index

def selection_sort(arr): Sorts an array
  result_array = []
  for i in range(len(arr)):
    smallest = find_smallest(arr)
    result_array.append(arr.pop(smallest))
  return result_array
```

### Divide & conquer
---
1. Figure out a simple case as the base case
2. Figure out how to reduce your problem and get to the base case
D&C works by breaking a problem down into smaller and smaller
pieces. If you’re using D&C on a list, the base case is probably an
empty array or an array with one element

### Quicksort
---
- Much faster than selection sort.
- Faster in practice because it hits the average case way more often than the worst case.
- Frequently used in real life
`[10, 5, 7, 33] --> [5, 7, 10], [33] // choose your pivot e.g. 33 and sort the others.`

e.g.:

```python
def quicksort(array):
  if len(array) < 2:
    return array Base case: arrays with 0 or 1 element are already “sorted.”
  else:
    pivot = array[0] Recursive case
    less = [i for i in array[1:] if i <= pivot] Sub-array of all the elements less than the pivot
    greater = [i for i in array[1:] if i > pivot] Sub-array of all the elements greater than the pivot
    return quicksort(less) + [pivot] + quicksort(greater)
```

### Dijkstra
Steps:
- Find the cheapest node
- Check the neighbors of this node to see if there is a cheapest path and update their costs
- Repeat for every node in the graph
- Calculate the final path

Only works with **positive weighted edges**


Example:
```python

def find_least_cost_node(costs):
    lowest_cost = infinity
    lowest_cost_node = None

    for each in costs:
        cost = costs[each]

        if cost < lowest_cost and each not in processed:
            lowest_cost_node = each
            lowest_cost = cost

    return lowest_cost_node

if __name__ == "__main__":
    infinity = float("inf")
    costs = {}
    graph = {}
    parents = {}
    processed = []

    graph["node1"] = ["node3"]
    graph["node2"] = ["node3", "node4"]
    graph["node3"] = ["node5"]
    graph["node5"] = ["node6"]
    graph["node4"] = ["node6"]

    graph["start"] = {}
    graph["start"]["node1"] = 2
    graph["start"]["node2"] = 5

    parents["node1"] = "start"
    parents["node2"] = "start"
    parents["end"] = None

    costs["node1"] = 2
    costs["node2"] = 2
    costs["node3"] = infinity
    costs["node4"] = infinity
    costs["node5"] = infinity
    costs["node6"] = infinity

    # given the initial costs the algorithm will choose between node1 and node2
    node = find_least_cost_node(costs)

    while node is not None:
        cost = costs[node]
        neighbors = graph[node]
        for each in neighbors:
            new_cost = cost + neighbors[each]
            if costs[each] > new_cost:
                costs[each] = new_cost
                parents[each] = node
        processed.append(node)
        node = find_least_cost_node(costs)

```


### Bellman-Ford

...


### Dynamic Programming
- Break a problem into discrete sub-problems
- Usefull when trying to optimize something given a constraint
- Every dynamic solution involves a grid with the cells being the sub-problems and what is needed to be optimized.
- There is no single formula to calculate a dynamic programming solution

todo: example


### Trees

\#todo

# B-trees

# Red-back trees

# Heaps

# Splay trees


...

### Fourier Transformation
```
"Given a smoothie, the Fourier transform will give you the ingredients of the smoothie"

 src: Better Explained
 ```



# Python 3 Features
`source: http://www.asmeurer.com/python3-presentation/slides.html#36`

### Feature 1: Advanced unpacking
```python
a, b, *rest = range(10) # will assign 0 to a, 1 to b and the rest to rest
a,*rest b,  = range(10) # will work too
*rest b,    = range(10) # will work too
with open("using_python_to_profit") as f:
  first, *_, last = f.readlines()

def f(*args):
    a, b, *args = args

```

### Feature 2: Keyword only arguments
```python
def f(a, b, *args, flag=True):
    # do something

# or
def f(a, b, *, flag=True):
    # do something that does not care about args
```

### Feature 3: Chained Exceptions
```python
try:
  do something
except IOError:
  raise NotImplementedError("Check for something") from IOError
```

### Feature 4: Fine grained OSError subclasses
from this:
```python
import errno

try:
    # do something that raises oserror
except OSError as e:
    if e.errno in [errno.EPERM, errno.EACCES]:
        raise NotImplementedError("automatic sudo injection")
    else:
        raise
```
to this:

```
try:
    # do something that raises a permission error
except PermissionError:
        raise NotImplementedError("automatic sudo injection")
```

### Feature 5: Everything is an iterator

No more xrange instead of range!

### Feature 6: Cannot compare everything to everything

```python
'one' > 2 # returns True in Python but this can yield unexpected results
```
This won't work in Python 3


### Feature 7: yield from

Instead of:
```python
for i in gen():
  yield i
```
do:
```python
yield from gen()
```

### Feature 8: asyncio
```python
# Taken from Guido's slides from “Tulip: Async I/O for Python 3” by Guido
# van Rossum, at LinkedIn, Mountain View, Jan 23, 2014
@coroutine
def fetch(host, port):
    r,w = yield from open_connection(host,port)
    w.write(b'GET /HTTP/1.0\r\n\r\n ')
    while (yield from r.readline()).decode('latin-1').strip():
        pass
    body=yield from r.read()
    return body

@coroutine
def start():
    data = yield from fetch('python.org', 80)
    print(data.decode('utf-8'))
```

### Feature 9: Standard library additions
faulthandler
ipaddress
functools.lru_cache
enum  # as it should be

### Feature 10: Unicode variable names
π = 3.14

Function annotations:
def some_f_that_returns_float() -> float:
  return 42.0

print(f.__annotations__)

### Feature 11: Unicode and bytes
At last `str` is a string and bytes are bytes.-

### Feature 12: Matrix Multiplication
np.dot(a, b) --> a @ b # override __matmul__ to use @

### Feature 13: Pathlib

Much simpler than `os. . . `:
```python
from pathlib import Path

directory = Path("/etc")
filepath = directory / "test_file.txt"

if filepath.exists():
    # do stuff
```


Python 3 - Async I/O
---
```python
import asyncio
import datetime
import random


@asyncio.coroutine
def display_date(num, loop):
    end_time = loop.time() + 50.0
    while True:
        print("Loop: {} Time: {}".format(num, datetime.datetime.now()))
        if (loop.time() + 1.0) >= end_time:
            break
        yield from asyncio.sleep(random.randint(0, 5))


loop = asyncio.get_event_loop()

asyncio.ensure_future(display_date(1, loop))
asyncio.ensure_future(display_date(2, loop))

loop.run_forever()
```

async/ await
---

```python
import asyncio
import datetime
import random


async def display_date(num, loop, ):
    end_time = loop.time() + 50.0
    while True:
        print("Loop: {} Time: {}".format(num, datetime.datetime.now()))
        if (loop.time() + 1.0) >= end_time:
            break
        await asyncio.sleep(random.randint(0, 5))


loop = asyncio.get_event_loop()

asyncio.ensure_future(display_date(1, loop))
asyncio.ensure_future(display_date(2, loop))

loop.run_forever()
```


# C# The Visitor pattern

[src](https://www.codeproject.com/Articles/588882/TheplusVisitorplusPatternplusExplained)


# Python desing patterns and idioms

[src1](https://www.toptal.com/python/python-design-patterns)

[src2](https://github.com/faif/python-patterns)

## The MVC Pattern
    [           Model            ]
        ^                   |
        |                   |
    Manipulates      Returns data
        |                   V
        [   Controller    ]
            ^           |
            |           |
            |        Updates / Renders
          Uses          |
            |           V
        [ User ]    [ View ]


- Model:
Responsible for maintaining the integrity of the data
Should not work directly wirh the view.
- View:
Responsible for the visualization of the data.
Should not access database directly or be heavy in logic.
- Controller:
The bridge between individual components of the system. Receives data from requests and sends it to other parts of the system.
Should not render data or work with the database and business logic directly.


## Singleton Pattern - Only one object
When:
- Need to control concurrent access to a shared resource, e.g. database connection
- Need a global point of access for the resource from multiple or different parts of the system.
- Need to have only one instance of an object.

Examples:
- The logging class and its subclasses
- Printer spooler
- Database connection
- File manager
- Retrieving and storing information using external configuration files
- Read only singletons for global states, e.g. user language

Module-level Singleton:
---
Check if module is already imported: import the initialized module and use it else find, initialize and return the module
Any access / import of the module after initialization will return the already initialized module.

```python
class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance
```

Identity comparison will return `True` when trying to compare two "instances" of the same Singleton, e.g.:
```python
>>> singleton_first = Singleton()
>>> singleton_second = Singleton()
>>> print(singleton_first is singleton_second)
>>> True
>>> class Child(Singleton):
 ...    pass
>>> child = Child()
>>> child is singleton_first
>>> False
```

The Borg singleton - monostate:
---
Shared state but **not** comparable through identity comparison `is`
```python
class Borg(object):
    _shared_state = {}
   
    def __new__(cls, *args, **kwargs):
        obj = super(Borg, cls).__new__(cls, *args, **kwargs)
        obj.__dict__ = cls._shared_state
        return obj

class Child(Borg):
    pass

>>> borg = Borg()
>>> borg.name = "Borg"
>>> child = Child()
>>> child.name
>>> Borg

# however:
class IndependentChild(Borg):
    _shared_state = {}

>>> independent_child = IndependentChild()
>>> independent_child.name
AttributeError: 'IndependentChild' object has no attribute 'name'
```
Note: Modules in Python are singleton by nature

## Factory Pattern
- Factories provide loose coupling, separating object creation and specific class implementation
- The Factory class can reuse existing objects.
- Direct instantiation always creates a new object
- The class that uses the new object does not need to know exaclty which class' instance was created, just the interface that is implemented by that class.
- To add a new class in a Factory: the class needs to implement the same interface.


```
[ Client ] ---------- [ Factory ]
    |                       |
    |_______________ [       Abstract class         ]
                        |       |    ....          |
                [ Implement.1 ][ Implement.2 ][ Implement.N ]
```



```
class ProtocolFactory(object):
    @staticmethod 
    def build_connection(protocol):
        if protocol == 'http':
            return HTTPConnection()
        elif protocol == 'ftp':
          return FTPConnection()
        raise RuntimeError('Unknown protocol')

if __name__ == '__main__':
    protocol = ProtocolFactory.build_connection('http)
    protocol.connect()
    print protocol.get_response()
```