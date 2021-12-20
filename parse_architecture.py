import xml.etree.ElementTree as ET
import pprint


def sum(a, b):
    return a+b


def parse(node, architecture):
    architecture[node.get('name')] = {'kind': node.tag,
                                      'children': [child.get('name') for child in node]
                                      }
    for child in node:
        architecture = parse(child, architecture)
    return architecture


tree = ET.parse('architecture.xml')
root = tree.getroot()
root.find('server')
pp = pprint.PrettyPrinter()
pp.pprint(parse(root.find('server'), {}))

fn = root.find('server').find('function').text.rstrip()
print(locals()[fn](3, 2))
