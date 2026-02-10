.. code-block:: python

  model = Modely()
  import pandas as pd
  import numpy as np

  df = pd.DataFrame({
      'time': np.array(range(60), dtype=np.float32),
      'x': np.array(10*[10] + 20*[20] + 30*[30], dtype=np.float32)
  })

  resampled_df = model.resamplingData(df, scale=1e9)
