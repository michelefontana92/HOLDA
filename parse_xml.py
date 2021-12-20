import functools
import importlib
import xml.etree.ElementTree as ET
from utils.read_xml import parse_architecture_tag, parse_function_tag, parse_metrics_tag, parse_task_tag, parse_setting_tag
import time

filename = 'test.xml'
tree = ET.parse(filename)

model = parse_function_tag(tree.find('model'), 'model_fn')
print('Model:\n', model)
print('Call Model fn: ', model())
metrics = parse_metrics_tag(tree)
print('Metrics:\n', metrics)
task = parse_task_tag(tree)
print('Task:\n', task)
setting = parse_setting_tag(tree)
print('Setting:\n', setting)

server = parse_architecture_tag(tree)
server.execute()
server.shutdown()
#s = 'RMSprop'
#m = importlib.import_module('torch.optim')
#print(functools.partial(getattr(m, s), lr=0.2))
