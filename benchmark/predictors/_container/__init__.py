"""Host-side glue for container-trained predictors (``ContainerPredictor``).

Everything the host does around a model ``.sif`` — build the combined-form
``config.json`` (:mod:`config`), launch the container (:mod:`runner`), validate
the config against the contract (:mod:`contract`), assert the split is leak-safe
(:mod:`leakage`), and map the container's ``predictions.h5ad`` onto the benchmark
delta tensor (:mod:`tensor_map`). Imported only by
``benchmark.predictors.container_predictor``.
"""
