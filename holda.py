import xml.etree.ElementTree as ET
from utils.read_xml import parse_architecture_tag
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'file', 'test.xml', 'path of the xml file where the architecture is described')


def main(argv):
    filename = FLAGS.file
    tree = ET.parse(filename)
    server = parse_architecture_tag(tree)
    server.execute()
    server.shutdown()


if __name__ == '__main__':
    app.run(main)
