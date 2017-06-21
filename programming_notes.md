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
