import xml.etree.ElementTree as ET
from utils.read_xml import parse_architecture_tag
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'file', 'test.xml', 'path of the xml file where the architecture is described')
flags.DEFINE_boolean(
    'only_pers', False, 'execute just a personalization')
flags.DEFINE_boolean(
    'pers', False, 'execute also the personalization step')


def main(argv):
    filename = FLAGS.file
    pers = FLAGS.pers
    only_pers = FLAGS.only_pers

    tree = ET.parse(filename)
    server = parse_architecture_tag(tree)
    if only_pers:
        server.personalize()
    else:
        server.execute()
        if pers:
            server.personalize()
    server.shutdown()


if __name__ == '__main__':
    app.run(main)
