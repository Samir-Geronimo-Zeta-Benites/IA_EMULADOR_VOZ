
import sys; sys.path.insert(0, '.')
from core.rvc_model.models import SynthesizerTrnMs256NSF
import inspect
sig = inspect.signature(SynthesizerTrnMs256NSF.__init__)
for name, param in sig.parameters.items():
    if name != 'self':
        d = param.default
        v = 'REQUIRED' if d is inspect.Parameter.empty else str(d)
        print(f'  {name}: {v}')

