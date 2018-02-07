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

Think about when you have to look up a name in the phonebook (age alert!!), if the name doesn't begin with A, but let's say
with K, then it makes more sense to open the phonebook somewhere in the middle. The same goes when looking up a word in a
dictionary and with when searching in a sorted list.

```python
number = 8
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
low = numbers[0]  # 0 is the lowest number in the list
high = numbers[-1]  # 9 is the highest number in the list
while low <=high:
    middle = len(numbers) / 2  # = 5
    guess = numbers[middle]
    is guess == number?   # 5 != 8
        yes: break and say we found the number in the list!
        no: continue
    is guess > number?    # 5 < 8
        yes: look in the first half of the list, numbers[:middle]
            high = middle - 1
        no: look in the second half of the list # this will be true
            high = middle + 1


```

### Traveling Salesman
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

# Python

## Context Managers
Two ways to write context managers:

```python
class ManagedFile(object):
    def __init__(self, name, handle='rb'):
        self.name =  name
        self.handle = handle
        
    def __enter__(self):
        self.file = open(self.name, self.handle)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

```

or:

```python
from contextlib import contextmanager

@contextmanager
def managed(name, handle='rb'):
    try:
        f = open(name, handle)
        yield f
    finally:
        f.close()    

```

A good example of usage would be to create an indenter:

```python

class Indenter(object):
    def __init__(self):
        self.level = 0
    
    def __enter__(self):
        self.level += 1
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.level -= 1
   
    def print_text(self, text):
        print ' ' * self.level + text


with Indenter as indent:
    indent.print_text('hi')
    with indent:
        indent.print_text('hi from second level')
indent.print_text('hi from unindented')

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
import abc


class Connection(object):
    __metaclass__ =  abc.ABCMeta
    
    def __init__():
        pass
    
    @abc.abstractmethod
    def get_response():
        pass

class HTTPConnection(Connection):
    def __init__():
        super(HTTPConnection, self).__init__()

    def get_response():
        # do something here to get a response
        return 'Hello from HTTP'


class FTPConnection(Connection):
    def __init__():
        super(FTPConnection, self).__init__()

    def get_response():
        # do something here to get a response
        return 'Hello from FTP'


class ProtocolFactory(object):
    @staticmethod 
    def build_connection(protocol):
        if protocol == 'http':
            return HTTPConnection()
        elif protocol == 'ftp':
          return FTPConnection()
        raise RuntimeError('Unknown protocol')

if __name__ == '__main__':
    protocol = ProtocolFactory.build_connection('http')
    protocol.connect()
    print protocol.get_response()
```



# What is the CAP theorem?
The CAP theorem states that in the presence of a network partition, one has to choose between consistency and availability.
In more detail:
The CAP theorem, or Brewer's theorem (Eric Brewerer) states that it is impossible for a distributed data store to simultaneously provide more than
two of the following three guarantees:

0. Consistency: Every read receives the most recent write or an error
1. Availability: Every request reveives a (non-error) response, without guarantee that it is the most recent write
2. Partition Tolerance: The system continues to operate despite an arbitrary number of messages being dropped (or delayed) by the network between the nodes.

The last one has to be tolerated in general, thus there is a choice between consistency and availability.

[src](https://en.wikipedia.org/wiki/CAP_theorem)


# ACID

## Atomicity
Each transaction will be all or nothing. If something goes wrong, the entire transaction fails (rollback)

## Consistency
The transaction will bring the database from one valid state to another, which means that any data written must follow the defined rules, for example,
the constraints, the cascade, the triggers and any combination of them.  Programming errors cannot result in violation in any of the aforementioned rules.

## Isolation
Ensures that the concurrent execution of transactions results in a system state that would be obtained if transactions where executed sequentially.

## Durability
Once a transaction has been committed, it will remain so, even in the event of a power loss, crashes and / or errors.

[src](https://en.wikipedia.org/wiki/ACID#Atomicity)


# General interview problems

[src1](http://www.practicepython.org/)

## Identify if anagram pair exists:

Given a list of words, return True if at least one anagram pair exists, False otherwise

```python
possible_anagrams_list =['aann', 'nana', 'abaa', 'nnn]

def has_anagram_couples(possible_anagrams_list):
    """
    :param: possible_anagrams_list list[str]:
    :return: boolean, True if at least one anagram pair exists, False otherwise
    """
    unique_ordered_words = set([''.join(ordered(w)) for w in possible_anagrams_list])

    return len(possible_anagrams_list) > len(unique_ordered_words)

```

## List Overlap

Take two lists, say for example these two:
```
a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
```
and write a program that returns a list that contains only the elements that are common 
between the lists (without duplicates). Make sure your program works on two lists of different sizes.

Extras:

Randomly generate two lists to test this
Write this in one line of Python (don’t worry if you can’t figure this out at this point - we’ll get to it soon)

```
def get_common_with_sets(list_a, list_b):
    """
    Returns a list with the common unique elements of the two lists
    :return: list[int]
    """
    return list(set(list_a).intersection(set(list_b)))

# or

def get_common_with_comprehension(list_a, list_b):
    """
    Returns a list with the common unique elements of the two lists
    :return: list[int]
    """
    return [i for i in list_a if i in list_b]


if __name__ == "__main__":
    from random import randint
    a = range(randint(0, 50))
    b = range(randint(12, 34))

    common_set = get_common_with_sets(a, b) 
    common_comprehension = get_common_with_comprehension(a, b)

    assert common_set == common_comprehension
    assert common_set == [i for i in a if i in b]
```

## Divisors

Create a program that asks the user for a number and then prints out a list of all the divisors of that number.

```python
num = None

while not isinstance(num, int):
    num = raw_input('Please, enter a number:')
    try:
        num = int(num)
    except ValueError:
        print 'Invalid value'

divisors = []
for i in range(1, num):
    if num % i == 0:
        divisors.append(i)

print 'The divisors are: {}'.format(','.join([str(d) for d in divisors]))

```

## String Lists
Ask the user for a string and print out whether this string is a palindrome or not. (A palindrome is a string that reads the same forwards and backwards.)

```python
def is_palindrome(possible_palindrome):
    return possible_palindrome[::-1] ==  possible_palindrome


possible_palindrome = raw_input('Please, enter a string:')
possible_palindrome = possible_palindrome.replace(' ','').lower()

while not possible_palindrome:
    print "Please enter more characters"
    possible_palindrome = raw_input('Please, enter a string:')
    possible_palindrome = possible_palindrome.replace(' ','').lower()

if is_palindrome(possible_palindrome):
    print "Is palindrome!"
else:
    print "Not a palindrome!"

```


## List Comprehensions
Let’s say I give you a list saved in a variable: `a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]`. 
Write one line of Python that takes this list a and makes a new list that has only the even elements of this list in it.

```python
a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
even_a = [num in a if num % 2 == 0]
```

## Rock Paper Scissors
Make a two-player Rock-Paper-Scissors game. 
(Hint: Ask for player plays (using input), compare them, print out a message of congratulations to the winner, and ask if the players want to start a new game)

Remember the rules:

Rock beats scissors
Scissors beats paper
Paper beats rock

```python
moves = {
    'rock': 'scissors',
    'paper': 'rock',
    'scissors': 'paper',
}

users = ['First', 'Second']
moves_list = ','.join(moves.keys())
input_template = '{} user, please enter your choice from ' + moves_list
verdict_template =  "{} player wins! {} beats {}"
# show the options
more_or_q = ''
while more_or_q != 'q':
    choices = []
    for user in users:
        valid = False
        while True:
            choice = raw_input(input_template.format(user)).lower()
            valid = choice in moves
            if not valid:
                print "Not a valid input, please choose from {}".format(moves_list)
            else:
                choices.append(choice)
                break
            
    a = choices[0]
    b = choices[1]
    if a == b:
        print "Tie!"
    # a beats b
    elif moves[a] == b:
        print verdict_template.format(users[0], a, b)
    else:  # a does not beat b
        print verdict_template.format(users[1], b, a)
    more_or_q = raw_input('Enter q to quit or anything else to continue playing:')
```

## Guessing game
Generate a random number between 1 and 9 (including 1 and 9). 
Ask the user to guess the number, then tell them whether they guessed too low, too high, or exactly right.

Extras:

- Keep the game going until the user types q
- Keep track of how many guesses the user has taken, and when the game ends, print this out.

```python
from random import randint

correct_guesses = 0
more_or_q = ''
choice = randint(1, 10)

print "Let's play a guessing game!\nI have chosen a number between 1 and 9, try to guess it!"

while more_or_q != 'q':
    
    while True:
        try:
            guess = int(raw_input("Enter your guess!"))
            break
        except ValueError:
            print "Not a number!"
    if choice == guess:
        correct_guesses += 1
        print "That's correct! The chosen number is {}".format(choice)
        # Choose a different number since the player guessed correctly
        choice = randint(1, 10)
    else:
        print "Unfortunatelly not correct. The number you guessed is {} than the number chosen.".format(
            "higher" if choice-guess < 0 else "lower"
            )
    more_or_q = raw_input('Enter q to quit or anything else to continue playing:')

print "Number of correct guesses: {}".format(correct_guesses)
```

## Check Primality Functions

```
Ask the user for a number and determine whether the number is prime or not. 
(A prime number is a number that has no divisors.)

```python
more_or_q = ''
while more_or_q != 'q':
    while True:
        try:
            guess = int(raw_input("Enter a number to check for primality!"))
            break
        except ValueError:
            print "Not a number!"

    if guess <=1:
        print "Not a prime number."
    is_prime = True

    for i in range(2, guess):
        if guess % i == 0:
            is_prime = False
            break

    print "It is a prime number!" if is_prime else "Not a prime number."
    more_or_q = raw_input('Enter q to quit or anything else to continue playing:')

```

## List Ends

Write a program that takes a list of numbers (for example, a = [5, 10, 15, 20, 25]) and 
makes a new list of only the first and last elements of the given list. 

```python

def first_and_last_list(input_list):
    if input_list:
        return [input_list[0], input_list[-1]] if len(input_list) > 1 else return [input_list[0]]
    return []

if __name__ == "__main__":
    a = [5, 10, 15, 20, 25]
    result = first_and_last_list(a)
    assert result == [5, 25]
    result = first_and_last_list([])
    assert result == []
    result = first_and_last_list([1])
    assert result == [1]

```

## Fibonacci

Write a program that asks the user how many Fibonnaci numbers to generate and then generates them. 
Make sure to ask the user to enter the number of numbers in the sequence to generate.
(Hint: The Fibonnaci seqence is a sequence of numbers where the next number in the sequence is the sum of the previous two numbers in the sequence. 
The sequence looks like this: 1, 1, 2, 3, 5, 8, 13, …)

```python
more_or_q = ''
while more_or_q != 'q':
    while True:
        try:
            num = int(raw_input("Enter the number of Fibonnaci numbers to generate:"))
            break
        except ValueError:
            print "Not a number!"
    fibonnaci = []
    for i in range(0, num + 1):
        if len(fibonnaci) > 1:
            fibonnaci.append(fibonnaci[i-2] + fibonnaci[i-1])
        else:
            fibonnaci.append(1)

    print "Your Fibonnaci sequence is: {}".format(",".join([str(f) for f in fibonnaci]))
    more_or_q = raw_input('Enter q to quit or anything else to continue playing:')
```

## Reverse Word Order
Write a program (using functions!) that asks the user for a long string containing multiple words. 
Print back to the user the same string, except with the words in backwards order. 
For example, say I type the string: `My name is Michele`
Then I would see the string:`Michele is name My` shown back to me.

```python
more_or_q = ''
while more_or_q != 'q':
    while True:
        words = raw_input("Enter the string of words to reverse:")
        if words:
            break
        else:
            print "Empty string."

    # doesn't take punctuation into account
    print " ".join(words.split(' ')[::-1])
    more_or_q = raw_input('Enter q to quit or anything else to try again:')
```

## Password Generator
Write a password generator in Python. Be creative with how you generate passwords - strong passwords have a mix of lowercase letters, 
uppercase letters, numbers, and symbols. 
The passwords should be random, generating a new password every time the user asks for a new password. 
Include your run-time code in a main method.

Extra:
- Ask the user how strong they want their password to be. For weak passwords, pick a word or two from a list.
```python
import random


# note: https://docs.python.org/2.7/library/os.html#os.urandom

def get_numeric(num_items):
    symbols = '0123456789'
    p = ''
    for i in range(0, num_items):
        p += random.choice(symbols)
    return p


def get_letters(num_items, upper_num=0):
    az = list('abcdefghijklmnopuvwxyz')
    p = []
    for i in range(0, num_items):
        p.append(random.choice(az))

    already_changed = []
    for i in range(0, upper_num):
        rand_position = random.randint(0, num_items-1)
        # do not change the same character again
        while rand_position in already_changed:
            rand_position = random.randint(0, num_items-1)
        p[rand_position] = p[rand_position].upper()
        already_changed.append(rand_position)
    return ''.join(p)


def get_symbols(num_items):
    symbols = '!@#$%^&*()_+\{\}[]|\:";\'<>,.?/'
    l = len(symbols)
    p = ''
    for i in range(0, num_items + 1):
        p += random.choice(symbols)
    return p


def get_password(strength):
    result = list(''.join(
        [f(*args) for f, args in password_strength[strength]])
    )
    random.shuffle(result)
    return ''.join(result)


password_strength = {
    'weak': {(get_letters, (5,))},
    'medium': {(get_letters, (6, 2)), (get_numeric, (2,))},
    'strong': {(get_letters, (8, 4)), (get_numeric, (2,)), (get_symbols, (2,))}
}


more_or_q = ''
while more_or_q != 'q':
    while True:
        strength = raw_input("What kind of password to create? (weak, medium, strong):")
        if strength:
            if strength in password_strength:
                break
            else:
                print "Not a valid option, please enter one of the following: weak, medium, strong"
        else:
            print "Empty string."
            
    print get_password(strength)
    more_or_q = raw_input('Enter q to quit or anything else to try again:')

```

## Decode A Web Page

Use the BeautifulSoup and requests Python packages to print out a list of all the article titles on the [NYTimes page](https://www.nytimes.com/)

```python
import requests
from bs4 import BeautifulSoup

r = requests.get('https://www.nytimes.com/')
r_html = r.text
soup = BeautifulSoup(r_html, 'html.parser')
for story_heading in soup.find_all(class_="story-heading"):
    if story_heading.a:
        print story_heading.a.text.replace("\n", " ").strip()
    else:
        print story_heading.contents[0].strip()
```

## Cows and Bulls
Create a program that will play the “cows and bulls” game with the user. The game works like this:

Randomly generate a 4-digit number. Ask the user to guess a 4-digit number. 
For every digit that the user guessed correctly in the correct place, they have a “cow”. 
For every digit the user guessed correctly in the wrong place is a “bull.” 
Every time the user makes a guess, tell them how many “cows” and “bulls” they have. 
Once the user guesses the correct number, the game is over. 
Keep track of the number of guesses the user makes throughout teh game and tell the user at the end.

Say the number generated by the computer is 1038. An example interaction could look like this:
```python
Welcome to the Cows and Bulls Game! 
Enter a number: 
>>> 1234
2 cows, 0 bulls
>>> 1256
1 cow, 1 bull
...
```
Until the user guesses the number.

```python
import random

all_nums = range(10)
more_or_q = ''
while more_or_q != 'q':
    all_cows = False
    cows = 0
    bulls = 0
    choice = [random.choice(all_nums) for i in range(4)]

    while not all_cows:
        nums = raw_input("Enter a 4-digit number:")
        if nums.isdigit():
            break
        else:
            print "Not a valid number."
        numns = int(nums)
        for i, each in enumerate(list(nums)):
            if choice[i] == each:
                cows += 1
            else:
                bulls += 1

    print "{} cow{}, {} bull{}".format(cows, "s" if cows > 1 else "", bulls,
                                       "s" if bulls > 1 else "")
    all_cows = bulls == 0
    if all_cows == 0:
        print "You win! Game over!"
        more_or_q = 'q'
    else:
        more_or_q = raw_input('Enter q to quit or anything else to try again:')
```

## Decode A Web Page Two

Using the requests and BeautifulSoup Python libraries, print to the screen the full text of the article 
on this website: http://www.vanityfair.com/society/2014/06/monica-lewinsky-humiliation-culture.
The article is long, so it is split up between 4 pages. 
Your task is to print out the text to the screen so that you can read the full article without having to click any buttons.
This will just print the full text of the article to the screen. 
It will not make it easy to read, so next exercise we will learn how to write this text to a .txt file.

```python
import requests
from bs4 import BeautifulSoup

url = 'http://www.vanityfair.com/society/2014/06/monica-lewinsky-humiliation-culture'
r = requests.get(url)
r_html = r.text
soup = BeautifulSoup(r_html, 'html.parser')
for section in soup.find_all(class_="content-section"):
        if section.p:
            print(section.p.text.replace("\n", " ").strip().encode('utf-8'))
```

## Element Search
Write a function that takes an ordered list of numbers (a list where the elements are in order from smallest to largest) 
and another number. The function decides whether or not the given number is inside the list and returns (then prints) 
an appropriate boolean.

Extras:
- Use binary search.

```python

def is_in_list(sorted_num_list, number):
    """
    Returns True if number in sorted_num_list, else False
    """

    low = 0
    high = len(sorted_num_list) - 1
    # check for early exits
    if number < sorted_num_list[low] or number > sorted_num_list[high]:
        return False
    if number == sorted_num_list[low] or number == sorted_num_list[high]:
        return True

    while low <= high:
        middle = (low + high) / 2
        item = sorted_num_list[middle]

        if item == number:
            return True
        if item > number:
            high = middle - 1
        else:
            low = middle + 1

    return False
```

## Write To A File
Take the code from the How To Decode A Website exercise
(if you didn’t do it or just want to play with some different code, use the code from the solution),
and instead of printing the results to a screen, write the results to a txt file.
In your code, just make up a name for the file you are saving to.

Extras:

- Ask the user to specify the name of the output file that will be saved.

```python
import requests
from bs4 import BeautifulSoup

def get_website_content_to_file(file_path="temp.txt"):
    url = 'http://www.vanityfair.com/society/2014/06/monica-lewinsky-humiliation-culture'
    r = requests.get(url)
    r_html = r.text
    soup = BeautifulSoup(r_html, 'html.parser')

    with open(file_path, 'wb') as mon_file:
        for section in soup.find_all(class_="content-section"):
            if section.p:
                mon_file.write(section.p.text.replace("\n", " ").strip().encode('utf-8'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", default='temp.txt')
    args = parser.parse_args()
    get_website_content_to_file(args.filepath)

or

filepath = raw_input('Please enter the full file path to save the content to:')
get_website_content_to_file(filepath)

```


## Read From File
Given a .txt file that has a list of a bunch of names, count how many of each name there are in the file,
and print out the results to the screen. I have a .txt file for you, if you want to use it!

Extra:

- Instead of using the .txt file from above (or instead of, if you want the challenge),
take this .txt file, and count how many of each “category” of each image there are.
This text file is actually a list of files corresponding to the SUN database scene recognition database,
and lists the file directory hierarchy for the images.
Once you take a look at the first line or two of the file, it will be clear which part represents the scene category.
To do this, you’re going to have to remember a bit about string parsing in Python 3. I talked a little bit about it in this post.


```python

def read_files():
    names_file = './data/test_names.txt'
    categories_file = './data/test_categories.txt'

    names = defaultdict(int)
    categories = defaultdict(int)

    with open(names_file) as nf:
        for line in nf.readlines():
            names[line.strip()] += 1

    print names

    with open(categories_file) as cf:
        for line in cf.readlines():
            split_line = line.strip().split('/')
            if len(split_line) > 4:
                c = '_'.join(split_line[2:4])
            else:
                c = split_line[2]
            categories[c] += 1

    print categories

```

## File Overlap
Given two .txt files that have lists of numbers in them, find the numbers that are overlapping.
One .txt file has a list of all prime numbers under 1000, and the other .txt file has a list of happy numbers up to 1000.

```python
def file_overlap():
    with open('./data/primenumbers.txt') as prime:
        primes = [int(n) for n in prime.readlines()]
        with open('./data/happynumbers.txt') as happyf:
            happy = [int(n) for n in happyf.readlines()]
            print [n for n in happy if n in primes]

```

## Draw A Game Board

Time for some fake graphics! Let’s say we want to draw game boards that look like this:
```
 --- --- ---
|   |   |   |
 --- --- ---
|   |   |   |
 --- --- ---
|   |   |   |
 --- --- ---
```
This one is 3x3 (like in tic tac toe). Obviously, they come in many other sizes (8x8 for chess, 19x19 for Go, and many more).

Ask the user what size game board they want to draw, and draw it for them to the screen using Python’s print statement.

```python
def get_board():
    board_dimensions = None

    while not board_dimensions:
        try:
            user_input = raw_input(
                "Please enter the board dimesions, width x height, separated by x, e.g. 3x3:")
            user_input = user_input.strip().lower()

            if 'x' not in user_input:
                raise ValueError('Not a valid input')
            if len(user_input) >= 3:
                board_dimensions = [int(x) for x in user_input.split('x')]
        except Exception as e:
            print e.message

    s = ' '
    n = '\n'
    v = '|' + 3 * s
    h = '----'

    board = ''

    for i in range(board_dimensions[1]):
        board += board_dimensions[0] * h
        board += n
        board += (board_dimensions[0] + 1) * v
        board += n
    # final line
    board += board_dimensions[0] * h
    board += n

    print board

```

## Guessing Game Two

You, the user, will have in your head a number between 0 and 100.
The program will guess a number, and you, the user, will say whether it is too high, too low, or your number.
At the end of this exchange, your program should print out how many guesses it took to get your number.
As the writer of this program, you will have to choose how your program will strategically guess.
A naive strategy can be to simply start the guessing at 1, and keep going (2, 3, 4, etc.) until you hit the number.
But that’s not an optimal guessing strategy.
An alternate strategy might be to guess 50 (right in the middle of the range),
and then increase / decrease by 1 as needed. After you’ve written the program,
try to find the optimal strategy!

```python
from random import randint


print "Pick a number from 0 to 100 and keep it in mind"
ready = raw_input("Press enter when ready.")
found = False
guess_list = range(0, 101)
guesses = []
guidance = ['y', 'lower', 'higher']

while not found:
    guess = randint(guess_list[0], guess_list[-1])
    guesses.append(guess)
    print "My guess is {}".format(guess)

    user_confirmation = raw_input(
        "If this is correct enter y else enter lower or higher to guide me")

    while user_confirmation not in guidance:
        user_confirmation = raw_input(
            "Not a valid option, please enter y if correct, else enter lower "
            "or higher to guide me"
        )

    if user_confirmation == 'y':
        found = True

    elif user_confirmation == 'lower':
        # narrow down the guess list to include only lower than the current
        # guess numbers
        guess_list = guess_list[:guess_list.index(guess)]

    elif user_confirmation == 'higher':
        # narrow down the guess list to include only higher than the current
        # guess numbers
        guess_list = guess_list[guess_list.index(guess)+1:]

print "It took me {} guesses to figure this out!".format(len(guesses))
```

```python
from random import randint


print "Pick a number from 0 to 100 and keep it in mind"
ready = raw_input("Press enter when ready.")
found = False
guess_list = range(0, 101)
guesses = []
guidance = ['y', 'lower', 'higher']

low = guess_list[0]
high = guess_list[-1]
user_confirmation = None

while low <= high and user_confirmation != 'y':
    # start with low and high?
    middle = (low + high) / 2
    # remove old guesses from guess_list
    # guess_list = list(set(guess_list).difference(set(guesses)))
    print guess_list
    guess = guess_list[middle]
    guesses.append(guess)
    print "My guess is {}".format(guess)

    user_confirmation = raw_input(
        "If this is correct enter y else enter lower or higher to guide me")

    while user_confirmation not in guidance:
        user_confirmation = raw_input(
            "Not a valid option, please enter y if correct, else enter lower "
            "or higher to guide me"
        )

    if user_confirmation == 'y':
        found = True

    elif user_confirmation == 'lower':
        # narrow down the guess list to include only lower than the current
        # guess numbers
        high = middle - 1

    elif user_confirmation == 'higher':
        # narrow down the guess list to include only higher than the current
        # guess numbers
        low = middle + 1

print "It took me {} guesses to figure this out!".format(len(guesses))

```

## Check Tic Tac Toe

If a game of Tic Tac Toe is represented as a list of lists, like so:

```python
game = [[1, 2, 0],
	    [2, 1, 0],
	    [2, 1, 1]]
```

where a 0 means an empty square, 
a 1 means that player 1 put their token in that space, 
and a 2 means that player 2 put their token in that space.
Given a 3 by 3 list of lists that represents a Tic Tac Toe game board, 
tell me whether anyone has won, 
and tell me which player won, if any. 
A Tic Tac Toe win is 3 in a row - either in a row, a column, or a diagonal.
Don’t worry about the case where TWO people have won - 
assume that in every board there will only be one winner.

```python
def generate_board(w, h):
    s = ' '
    n = '\n'
    vl = '|' + 3 * s
    hl = '----'

    board = ''

    for i in range(h):
        board += w * hl
        board += n
        board += (w + 1) * vl
        board += n
    # final line
    board += w * vl
    board += n

    print board


def generate_random_matrices(w, h, low=0, high=2):
    from random import randint
    matrix = [[randint(low, high) for i in range(w)] for _ in range(h)]

    return matrix


def get_diagonals(matrix):
    m_len = len(matrix)

    d1 = [matrix[i][i] for i in range(m_len)]
    d2 = [matrix[i][i] for i in range(m_len)[::-1]]

    return d1, d2


def get_transposition(matrix):
    """
    [[ 0, 0, 0],   T   [[0, 1, 4],
     [ 1, 2, 3],  -->   [0, 2, 4],
     [4, 4, 4]]         [0, 3, 4]]
    :param matrix: an orthogonal matrix
    :return: list[list[T]]: the transposed matrix
    """
    if len(matrix[0]) != len(matrix):
        raise ValueError("Not an orthogonal matrix.")
    T = [range(len(matrix)) for _ in range(len(matrix))]

    for row_num, line in enumerate(matrix):
        for i, n in enumerate(line):
            T[i][row_num] = n

    return T


def assess_winner(matrix):
    """
    Checks if user 1 or 2 has won: 3 consecutive 1s or 2s either vertically,
    or horizontally or diagonally
    :param matrix: a 3 by 3 matrix
    :return: int, the number of the winner or 0 if no one won.
    """
    winning_rules = {
        1: [1] * 3,
        2: [2] * 3
    }

    # accumulate all cases: columns, rows, diagonals
    matrix_t = get_transposition(matrix)
    diagonals = get_diagonals(matrix)
    matrices = [matrix, matrix_t, diagonals]
    # check rows, columns and diagonals
    for matrix in matrices:
        for row in matrix:
            for k, v in winning_rules.iteritems():
                if row == v:
                    return k
    return 0


if __name__ == '__main__':
    # result = assess_winner(generate_random_matrices(3, 3))
    # if result == 0:
    #     print "No player won!"
    # else:
    #     print "Player no {} won!".format(result)
    #
    # game = [[1, 2, 0],
    #         [2, 1, 0],
    #         [2, 1, 1]]
    # assert assess_winner(game) == 1

    diagonal_m = [[1, 0, 2],
                  [0, 1, 2],
                  [0, 2, 1]]

    winner_is_2 = [[2, 2, 0],
                   [2, 1, 0],
                   [2, 1, 1]]
    winner_is_1 = [[1, 2, 0],
                   [2, 1, 0],
                   [2, 1, 1]]
    winner_is_also_1 = [[0, 1, 0],
                        [2, 1, 0],
                        [2, 1, 1]]
    no_winner = [[1, 2, 0],
                 [2, 1, 0],
                 [2, 1, 2]]
    also_no_winner = [[1, 2, 0],
                      [2, 1, 0],
                      [2, 1, 0]]

    assert assess_winner(winner_is_2) == 2
    assert assess_winner(winner_is_1) == 1
    assert assess_winner(winner_is_also_1) == 1
    assert assess_winner(no_winner) == 0
    assert assess_winner(also_no_winner) == 0
    assert assess_winner(diagonal_m) == 1

```

## Tic Tac Toe Draw

When a player (say player 1, who is X) wants to place an X on the screen, 
they can’t just click on a terminal. So we are going to approximate this 
clicking simply by asking the user for a coordinate of where they want to place their piece.

As a reminder, our tic tac toe game is really a list of lists. 
The game starts out with an empty game board like this:

```
game = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]
```
The computer asks Player 1 (X) what their move is (in the format row,col), 
and say they type 1,3. Then the game would print out

```
game = [[0, 0, X],
        [0, 0, 0],
        [0, 0, 0]]
```

And ask Player 2 for their move, printing an O in that place.

Things to note:

- Assume that player 1 (the first player to move) will always be X and player 2 
(the second player) will always be O.
- Notice how in the example I gave coordinates for where I want to move starting 
from (1, 1) instead of (0, 0). To people who don’t program, starting to count 
at 0 is a strange concept, so it is better for the user experience if the row 
counts and column counts start at 1. This is not required, but whichever way 
you choose to implement this, it should be explained to the player.

- Ask the user to enter coordinates in the form “row,col” - a number, then a 
comma, then a number. Then you can use your Python skills to figure out which 
row and column they want their piece to be in.
- Don’t worry about checking whether someone won the game, but if a player 
tries to put a piece in a game position where there already is another piece, 
do not allow the piece to go there.
Bonus:
- For the “standard” exercise, don’t worry about “ending” the game - no need to 
keep track of how many squares are full. In a bonus version, keep track of how 
many squares are full and automatically stop asking for moves when there are no 
more valid moves.


```python
from collections import OrderedDict


def get_diagonals(matrix):
    m_len = len(matrix)

    d1 = [matrix[i][i] for i in range(m_len)]
    d2 = [matrix[i][i] for i in range(m_len)[::-1]]

    return d1, d2


def get_transposition(matrix):
    """
    [[0, 0, 0],   T   [[0, 1, 4],
     [1, 2, 3],  -->   [0, 2, 4],
     [4, 4, 4]]         [0, 3, 4]]
    :param matrix: an orthogonal matrix
    :return: list[list[T]]: the transposed matrix
    """
    if len(matrix[0]) != len(matrix):
        raise ValueError("Not an orthogonal matrix.")
    T = [range(len(matrix)) for _ in range(len(matrix))]

    for row_num, line in enumerate(matrix):
        for i, n in enumerate(line):
            T[i][row_num] = n

    return T


def get_orthogonal_matrix(n, v='-'):
    """
    Returns an orthogonal matrix, e.g. if n is 3
    then the result will be:
    [[v, v, v],
     [v, v, v],
     [v, v, v]]
    :param n: number for the size of the matrix
    :param v: the default value to initialize the matrix with
    :return: list[list[T]]
    """
    return [[v for _ in range(n)] for i in range(n)]


def get_stringified_board(matrix):
    """
    Prints a matrix in a tabular, readable form
    :param matrix:
    :return:
    """
    v = '|'
    h = '----'
    s = ' '
    n = '\n'

    board = ''
    for row in matrix:
        board += h * len(row) + n
        for item in row:
            board += v + s + item + s
        board += v
        board += n
    board += h * len(matrix[0]) + n



def assess_winner(matrix, p1_pawn='X', p2_pawn='O'):
    """
    Checks if player 1 or 2 has won: 3 consecutive pawns either vertically,
    horizontally or diagonally
    :param list[list[T] matrix: a 3 by 3 matrix
    :param str p1_pawn: the pawn of player 1
    :param str p2_pawn: the pawn of player 1
    :return: int, the number of the winner or 0 if no one won.
    """
    winning_rules = {
        1: [p1_pawn] * 3,
        2: [p2_pawn] * 3
    }

    # accumulate all cases: columns, rows, diagonals
    matrix_t = get_transposition(matrix)
    diagonals = get_diagonals(matrix)
    matrices = [matrix, matrix_t, diagonals]
    # check rows, columns and diagonals
    for matrix in matrices:
        for row in matrix:
            for k, v in winning_rules.iteritems():
                if row == v:
                    return k
    return 0


def no_more_moves(matrix, pawns):
    # or all([row.count('-') == 0 for row in matrix])
    return list(set([item for row in matrix for item in row])) == all(pawns)


def is_move_valid(x, y, matrix, marker, empty='-'):
    l = len(matrix) - 1
    if x < 0 or y < 0 or x > l or y > l:
        return False

    if matrix[x][y] != empty or matrix[x][y] == marker:
        return False

    return True


def is_input_num_tuple(str_input):
    try:
        str_input = str_input.strip().replace(' ', '').split(',')
        x, y = str_input
        print "x {} y {}".format(x, y)
        return int(x), int(y)
    except:
        return None


if __name__ == '__main__':
    
    valid_pawns = ['X', 'O']
    
    print "This is a game of Tic Tac Toe."
    matrix = get_orthogonal_matrix(3)
    print matrix
    print get_stringified_board(matrix)
    
    pawn_player_1 = None
    x, y = None, None
    
    while pawn_player_1 not in valid_pawns:
        pawn_player_1 = raw_input("Player 1: Please select your pawn: X or O") \
            .upper()
    
    pawn_player_2 = valid_pawns[0] \
        if valid_pawns.index(pawn_player_1) == 1 else valid_pawns[1]
    
    # ordered to have the players turns right
    players_to_pawns = OrderedDict()
    players_to_pawns["Player 1"] = pawn_player_1
    players_to_pawns["Player 2"] = pawn_player_2
    
    victory_or_done = False
    
    while not victory_or_done:
        for p in players_to_pawns.keys():
            current_pawn = players_to_pawns[p]
            print "Currently playing with {}".format(current_pawn)
            while True:
                move_player_1 = raw_input(
                    "{}: Please enter your move in form of row, column, e.g. 1, 3"
                        .format(p))
    
                coords = is_input_num_tuple(move_player_1)
    
                while coords is None:
                    try:
                        move_player_1 = raw_input(
                            "{}: Not a valid input, please try "
                            "again e.g. 1, 3")
                        coords = is_input_num_tuple(move_player_1)
                        print coords
    
                    except:
                        print "Not a valid option."
    
                x, y = coords
                actual_x = x - 1
                actual_y = y - 1
    
                if is_move_valid(actual_x, actual_y, matrix, current_pawn):
                    matrix[actual_x][actual_y] = current_pawn
                    break
                else:
                    print "Not a valid option."
    
            victory_or_done = assess_winner(matrix) in [1, 2] or \
                              no_more_moves(matrix, valid_pawns)
    
            print get_stringified_board(matrix)
    
            if victory_or_done:
                print "{} wins!!!".format(p)
                break
```

## Max Of Three
Implement a function that takes as input three variables, and returns the largest of the three. 
Do this without using the Python max() function!

The goal of this exercise is to think about some internals that Python normally 
takes care of for us. All you need is some variables and if statements!

```python
def get_max_of_three(a, b ,c):
    return sorted(set([a, b, c]))[-1]

```

## Pick Word
The task is to write a function that picks a random word from a list of words 
from the SOWPODS dictionary. Download this file and save it in the same directory 
as your Python code. 
This file is Peter Norvig’s compilation of the dictionary of words used in 
professional Scrabble tournaments. Each line in the file contains a single word.


```python
word_list = []
with open('./data/sowpods.txt') as sowpods_f:
    for line in sowpods_f.readlines():
        word_list.append(line.strip())
    
from random import randint

print "Picked {} ".format(word_list[randint(0, len(word_list))])

```

## Guess Letters

In the game of Hangman, a clue word is given by the program that the player has 
to guess, letter by letter. The player guesses one letter at a time until the 
entire word has been guessed. 
(In the actual game, the player can only guess 6 letters incorrectly before losing).

Let’s say the word the player has to guess is “EVAPORATE”. 
For this exercise, write the logic that asks a player to guess a letter and displays 
letters in the clue word that were guessed correctly. 
For now, let the player guess an infinite number of times until they get the 
entire word. As a bonus, keep track of the letters the player guessed and display 
a different message if the player tries to guess that letter again. 
Remember to stop the game when all the letters have been guessed correctly! 
Don’t worry about choosing a word randomly or keeping track of the number of 
guesses the player has remaining - we will deal with those in a future exercise.

An example interaction can look like this:

>>> Welcome to Hangman!
_ _ _ _ _ _ _ _ _
>>> Guess your letter: S
Incorrect!
>>> Guess your letter: E
E _ _ _ _ _ _ _ E
...



----------------
[Other src](https://github.com/zhiwehu/Python-programming-exercises/blob/master/100%2B%20Python%20challenging%20programming%20exercises.txt)
## Question 2:

Write a program which can compute the factorial of a given numbers.
The results should be printed in a comma-separated sequence on a single line.
Suppose the following input is supplied to the program:
8
Then, the output should be:
40320

Factorial: the product of an integer and all the integers below it; e.g., factorial four ( 4! ) is equal to 24.

```python

def factorial(num):
    if num == 1:
        return num
    return num * factorial(num-1)
```

## Question 3:

With a given integral number n, write a program to generate a dictionary that 
contains (i, i*i) such that is an integral number between 1 and n (both included). 
And then the program should print the dictionary.
Suppose the following input is supplied to the program:
8
Then, the output should be:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
Consider use dict()

```python

def get_dict(num):
    return {i: i*i for i in range(1, num+1)}

```

## Question 6
Level 2

Question:
Write a program that calculates and prints the value according to the given formula:
Q = Square root of [(2 * C * D)/H]
Following are the fixed values of C and H:
C is 50. H is 30.
D is the variable whose values should be input to your program in a comma-separated sequence.
Example
Let us assume the following comma separated input sequence is given to the program:
100,150,180
The output of the program should be:
18,22,24

```python
def get_result(D):
    import math
    C = 50
    H = 30
    return round(math.sqrt((2*C*D) / H))
    
for i in range(3):
    while True:
        try:
            D = int(raw_input("Please enter number {}".format(i + 1)))
            print get_result(D)
            break
        except:
            print "Value error."
        
```

Question 19
Level 3

Question:
You are required to write a program to sort the (name, age, height) tuples by 
ascending order where name is string, age and height are numbers. 
The tuples are input by console. The sort criteria is:
1: Sort based on name;
2: Then sort based on age;
3: Then sort by score.
The priority is that name > age > score.
If the following tuples are given as input to the program:
Tom,19,80
John,20,90
Jony,17,91
Jony,17,93
Json,21,85
Then, the output of the program should be:
[('John', '20', '90'), ('Jony', '17', '91'), ('Jony', '17', '93'), ('Json', '21', '85'), ('Tom', '19', '80')]

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
We use itemgetter to enable multiple sort keys.

```python
from operator import itemgetter

inventory = []
while True:
    name_age_height = raw_input("Enter name, age, height or q to quit")
    if name_age_height == 'q':
        break
    if not name_age_height:
        print "No input"
    else:
        try:
            name, age, height = name_age_height
            inventory.append((name.strip(), age.strip(), height.strip()))
        except:
            print "Wrong values"

print sorted(inventory, itemgetter(0,1,2))
```


## Question 20
Level 3

Question:
Define a class with a generator which can iterate the numbers, which are divisible by 7, between a given range 0 and n.

```python
def divisible_by_seven_gen(n):
    for i in range(0, n):
        if i % 7 == 0:
            yield i

```

## product
You are given a two lists  and . Your task is to compute their cartesian product X.
[other src](https://www.hackerrank.com/challenges/itertools-product/problem)

Example

A = [1, 2]
B = [3, 4]

AxB = [(1, 3), (1, 4), (2, 3), (2, 4)]
Note:  and  are sorted lists, and the cartesian product's tuples should be output in sorted order.

Input Format

The first line contains the space separated elements of list . 
The second line contains the space separated elements of list .

Both lists have no duplicate integer elements.

Constraints

 

Output Format

Output the space separated tuples of the cartesian product.

Sample Input

 1 2
 3 4
Sample Output

 (1, 3) (1, 4) (2, 3) (2, 4)
 
 
```python
A = raw_input()
B = raw_input()

def get_valid_input(input_str):
    return [int(i.strip()) for i in input_str.split()]

A = get_valid_input(A)
B = get_valid_input(B)

def get_product_of(A, B):
    from itertools import product
    return ' '.join([str(a) for a in product(A, B)])

print get_product_of(A, B)
```


## permutations

You are given a string . 
Your task is to print all possible permutations of size  of the string in lexicographic sorted order.

Input Format

A single line containing the space separated string  and the integer value .

Constraints

 
The string contains only UPPERCASE characters.

Output Format

Print the permutations of the string  on separate lines.

Sample Input

HACK 2
Sample Output

AC
AH
AK
CA
CH
CK
HA
HC
HK
KA
KC
KH

```python
def get_string_uppercase_characters(str_input):
    ''.join([c for c in str_input.split() if c.isupper() and c.isalpha()])


def get_str_and_perm_nums(str_input):
    split_str = str_input.strip().split()
    return ''.join(split_str[:-1]), int(split_str[-1])


def get_permutations(capitalized_str, num_perm):
    from itertools import permutations
    return sorted([''.join(p) for p in permutations(capitalized_str, num_perm)])


str_input_perm = raw_input()

string_to_permutate, num_permutations = get_str_and_perm_nums(str_input_perm)

for p in get_permutations(string_to_permutate, num_permutations):
    print p
    
```

## combinations
You are given a string . 
Your task is to print all possible combinations, up to size , of the string in lexicographic sorted order.
Input Format
A single line containing the string  and integer value  separated by a space.

Constraints: 
The string contains only UPPERCASE characters.

Output Format
Print the different combinations of string  on separate lines.

Sample Input
HACK 2
Sample Output
A
C
H
K
AC
AH
AK
CH
CK
HK

```python
def get_string_uppercase_characters(str_input):
    ''.join([c for c in str_input.split() if c.isupper() and c.isalpha()])


def get_str_and_comb_nums(str_input):
    split_str = str_input.strip().split()
    return ''.join(split_str[:-1]), int(split_str[-1])


def get_combinations(capitalized_str, num_comb):
    from itertools import combinations
    return sorted([''.join(p) for p in combinations(capitalized_str, num_comb)])


str_input_comb = raw_input()

string_to_combine, num_combinations = get_str_and_comb_nums(str_input_comb)

for i in range(1, num_combinations + 1):
    for p in get_combinations(string_to_combine, i):
        print p

```


## Compress the string
Sample Input
1222311
Sample Output
(1, 1) (3, 2) (1, 3) (2, 1)

```python
input_str = raw_input()


def compress_string(string_to_compress):
    result = []
    current_char = string_to_compress[0]
    current_count = 1
    get_output = lambda cc, ci: str((cc, int(ci)))

    for i in range(1, len(string_to_compress)):
        seen_before = string_to_compress[i] == string_to_compress[i-1]
        if not seen_before:
            result.append(get_output(current_count, current_char))
            current_char = string_to_compress[i]
            current_count = 1
        else:
            current_count += 1

        if i == len(string_to_compress)-1:
            result.append(get_output(current_count, current_char))

    return ' '.join(result)


print compress_string(input_str)
```

## Maximize it [src](https://www.hackerrank.com/challenges/maximize-it/problem)
You are given a function . You are also given  lists. The  list consists of  elements.

You have to pick one element from each list so that the value from the equation below is maximized: 

Note that you need to take exactly one element from each list, not necessarily the largest element. 
You add the squares of the chosen elements and perform the modulo operation. 
The maximum value that you can obtain, will be the answer to the problem.

Input Format
The first line contains  space separated integers  and . 
The next  lines each contains an integer , denoting the number of elements in the  
list, followed by space separated integers denoting the elements in the list.

Output Format

Output a single integer denoting the value .

Sample Input

3 1000
2 5 4
3 7 8 9 
5 5 7 8 9 10 
Sample Output

206


## Deque

A deque is a double-ended queue. It can be used to add or remove elements from both ends.

Deques support thread safe, memory efficient appends and pops from either side of the deque with approximately the same O(1) 
performance in either direction.


```python
>>> from collections import deque
>>> d = deque()
>>> d.append(1)
>>> print d
deque([1])
>>> d.appendleft(2)
>>> print d
deque([2, 1])
>>> d.clear()
>>> print d
deque([])
>>> d.extend('1')
>>> print d
deque(['1'])
>>> d.extendleft('234')
>>> print d
deque(['4', '3', '2', '1'])
>>> d.count('1')
1
>>> d.pop()
'1'
>>> print d
deque(['4', '3', '2'])
>>> d.popleft()
'4'
>>> print d
deque(['3', '2'])
>>> d.extend('7896')
>>> print d
deque(['3', '2', '7', '8', '9', '6'])
>>> d.remove('2')
>>> print d
deque(['3', '7', '8', '9', '6'])
>>> d.reverse()
>>> print d
deque(['6', '9', '8', '7', '3'])
>>> d.rotate(3)
>>> print d
deque(['8', '7', '3', '6', '9']
```