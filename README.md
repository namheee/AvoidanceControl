# AvoidanceControl
Attractor avoidance control algorithm

## Usage


## Notes
+ To generate random Boolean networks, we utilized [BNGenerator](https://github.com/choonlog/OutputStabilization), a software tool that constructs random networks based on biological Boolean logics extracted from the Cell Collective (https://cellcollective.org/).
+ To identify all attractors of Boolean networks, we utilized [BooleanSim](https://github.com/jehoons/BooleanSim), a Python 3 tool developed based upon the [booleannet](https://github.com/ialbert/booleannet).


## Comparison with other algorithms
+ For brute force search and LDOI control, we utilited [pystablemotifs](https://github.com/jcrozum/pystablemotifs).
```python
pystablemotifs.drivers.minimal_drivers for brute force search

pystablemotifs.drivers.GRASP(GRASP_iterations = 2,000) for LDOI control
```

+ For [stable motif control](https://github.com/jcrozum/pystablemotifs),
```python
ar=pystablemotifs.AttractorRepertorie.from_primes
ar.succession_diagram.reprogram_to_trap_spaces()
```

+ For [FVS control](https://github.com/CASCI-lab/CANA),
```python
cana.control.feedback_vertex_set_driver_nodes(graph='structural', method='bruteforce', max_search=10, keep_self_loops=True)
```

+ For [biobalm control](https://github.com/jcrozum/biobalm)
```python
biobalm.SuccessionDiagram.from_rules and biobalm.control.succession_control()
```


## Reference paper
+ A related paper is expected to be published soon.
