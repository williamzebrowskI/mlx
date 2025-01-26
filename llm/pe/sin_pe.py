from pydantic import BaseModel
from mlx.core import sin, cos



class PositionalEmbeddings:
        def __init__(self, l, d):

            pass

        def run(self, input_seq):
              
              for i in input_seq:
                    x = sin(i)
                    x = cos(i)
                    return x


