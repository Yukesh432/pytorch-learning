class rectangle:
    def __init__(self, width, height):
        self._width= width
        self._height= height

    @property
    def width(self):
        """get the width of rectangle"""
        return self._width
    
    # width.setter
    def width(self, value):
        """Set the width of the rectangle"""
        if value<0:
            raise ValueError("Width cannot be negative.")
        self._width= value

    @width.deleter
    def width(self):
        """Delete the width of the rectangle"""
        del self._width

#creating an instance of Rectangle
rect= rectangle(12, 3)

#get the width
print(rect.width)
#set a new width
rect.width=5
print(rect.width)

print(rect.width)

