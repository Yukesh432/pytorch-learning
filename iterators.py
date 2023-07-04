"""
The iter() and next() function are used to iterate over iterable objects in python. An iterable object is an object
that can be looped over, such as a list, a tuple, or a string.

The iter() function takes an iterable object as input and returns an iterator object. The iterator object can then
be used to iterate over the iterable object. The next() function takes an iterator object as input and returns the
next element in the iterator object. If there are no more elements in the iterator object, the next() function will 
raise a StopIteration exception.

Are there any alternative of Iterator??

There are other ways to iterate over iterable objects, such as using the for loop. However, the iter() function has 
several advantages over the for loop. First, the iter() function allows us to create iterators for any iterable object, 
while the for loop only works for lists, tuples, and strings. Second, the iter() function is more efficient than the for loop, 
especially when iterating over large iterable objects.

"""

# list_= [1,2,3]

# iterator= iter(list_)

# while True:
#     try:
#         element= next(iterator)
#         print(element)
#     except StopIteration:
#         break
# ................................................................
# string= "hello"

# iterator= iter(string)

# xx= iter(iterator)
# yy= next(xx)

# print(xx)
# print(yy)

strings = ["hello", "world", "python"]

strings.reverse()
iterator = iter(strings)

while True:
    try:
        character = next(iterator)
        print(character, end=" ")
    except StopIteration:
        break