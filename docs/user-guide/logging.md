# Demonstrate usage of logging

PyPSA uses the Python standard library [logging](https://docs.python.org/3/library/logging.html).
This notebook shows how to use it and control the logging messages from different modules.

One can set the logging level to different values like **ERROR**, **WARNING**, **INFO**, **DEBUG**. This works independently for separate module. 

We start by setting the basic logging level to **ERROR**. 

```python
import logging

import pypsa

logging.basicConfig(level=logging.ERROR)
```

```python
network = pypsa.examples.ac_dc_meshed(from_master=True)
```

```python
out = network.optimize()
```

Now turn on infos just for optimization module.

```python
pypsa.optimization.optimize.logger.setLevel(logging.INFO)
```

```python
out = network.optimize()
```

Now turn on warnings just for optimization module

```python
pypsa.optimization.optimize.logger.setLevel(logging.WARNING)
```

```python
out = network.optimize()
```

Now turn on all messages for the PF module

```python
pypsa.pf.logger.setLevel(logging.DEBUG)
```

```python
out = network.lpf()
```

Now turn off all messages for the PF module again

```python
pypsa.pf.logger.setLevel(logging.ERROR)
```

```python
out = network.lpf()
``` 