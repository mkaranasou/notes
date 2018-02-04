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

```
number = 8
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
low = numbers[0]  # 0 is the lowest number in the list
high = numbers[-1]  # 9 is the highest number in the list
while low <=high:
    middle = len(numbers) / 2 = 5
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


```