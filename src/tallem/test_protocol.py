from typing import Protocol, runtime_checkable

@runtime_checkable
class Portable(Protocol):
	def port(): 
		print("yes")

class Mug:
	def __init__(self) -> None:
		self.handles = 1


if __name__ == '__main__':
	mug = Mug()
	print(isinstance(mug, Portable))