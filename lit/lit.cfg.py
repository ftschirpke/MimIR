import lit.formats
import shutil
import os

config.name = 'mim regression'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mim']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root, 'test')

config.substitutions.append(('%mim', config.mim))

acpp_lib_dir = getattr(config, 'acpp_lib_dir', '')
if acpp_lib_dir:
    config.substitutions.append(
        ('%acpp_link',
         '-L{0} -lacpp-rt -lacpp-common -Wl,-rpath={0} -pthread -ldl'.format(acpp_lib_dir)))
    config.available_features.add("pcuda")

if shutil.which("nvidia-smi") is not None:
    config.available_features.add("nvptx")

# inhert env vars
config.environment = os.environ

config.available_features.add("always")
