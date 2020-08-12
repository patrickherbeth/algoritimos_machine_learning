import os

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petrinet import visualizer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Algoritimo Alpha

log = xes_importer.apply('sample_data/running-example.xes')
net, initial_marking, final_marking = alpha_miner.apply(log)

# Visualização

alpha_miner.apply(log)
gviz = visualizer.apply(net, initial_marking, final_marking)
visualizer.view(gviz)


