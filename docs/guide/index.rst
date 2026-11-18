User Guide
==========

The guide follows the life of a model: how it is described, fed with data,
trained, validated, extended over time, and finally reused or exported.

Every page has the same plan:

- what you will be able to do at the end, and the pages to read first;
- one example system, used in every section of the page;
- numbered steps, each with a short explanation, the code and what it prints;
- a table of the common errors, with their cause and their fix;
- links to the reference pages of the names it uses.

Every page is a script: its code blocks, run from top to bottom, work as they
are, and print what the page shows. A test runs them on every backend. The
few blocks that need something optional, such as ONNX, say so.

The :doc:`tutorials <../tutorial/index>` teach the same topics as Jupyter
notebooks, and the :doc:`../glossary` defines the terms.

.. toctree::
   :maxdepth: 2

   concepts
   models
   data
   training
   validation
   recurrent
   physics
   composition
   export
