.. _nnodely-dataset-creation:

Dataset Creation
========================
Training data is a critical channel for domain knowledge. Selecting representative samples, shaping their distribution in the input space, and applying physics-informed data augmentation techniques can significantly affect model generalization. Conversely, biased or uninformative data may encode incorrect assumptions about the underlying system dynamics.

In nnodely, dataset creation is designed to preserve strict consistency between data representation and the temporal structure of the neural model.

Once loaded, via the :func:`loadData() <nnodely.operators.loader.Loader.loadData>` function, the datasets are automatically integrated according to the defined Inputs and their associated temporal windows. This guarantees that the temporal context required by the model is preserved, enabling correct construction of input sequences for time-dependent architectures.

.. rubric:: Multi-Source Support and Temporal Operations
The framework supports multiple data sources and provides utilities for :func:`resampling <nnodely.operators.loader.Loader.resamplingData>`, such as interpolation, as well as utilities for extracting specific temporal intervals. These features enable controlled experimentation under different operating conditions while maintaining alignment with the model's temporal structure.

.. rubric:: Multi-File handling
The framework also supports a multi-file dataset mode, where a directory of data files is treated as a single logical dataset. Data from different files are processed independently and concatenated while preserving temporal coherence, ensuring that valid temporal windows are constructed separately for each sequence. This capability is essential for recurrent training and closed-loop prediction scenarios, where temporal consistency across multiple trajectories must be strictly maintained.

.. .. rubric:: Key Benefits
.. .. ----------------
.. - Ensures that the temporal context required by the model is preserved during preprocessing.

.. - Facilitates reproducible and controlled experiments across different operating conditions.

.. - Enables seamless integration of heterogeneous data sources without compromising temporal coherence.

.. - Supports recurrent and closed-loop scenarios through multi-file dataset handling.

.. toctree::
   :maxdepth: 1

   loader_module
