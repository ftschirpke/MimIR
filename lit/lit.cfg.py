import lit.formats
import os

config.name = 'mim regression'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mim']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root, 'test')

config.substitutions.append(('%mim', config.mim))

# Link recipe for pCUDA-backend host modules against the AdaptiveCpp SSCP
# runtime (libacpp-rt / libacpp-common). Mirrors the recipe the pCUDA emitter
# prints. config.acpp_lib_dir is baked in by CMake's find_package(AdaptiveCpp);
# it is empty when AdaptiveCpp was not found, in which case the 'pcuda' feature
# below is not advertised and REQUIRES: pcuda tests are skipped.
acpp_lib_dir = getattr(config, 'acpp_lib_dir', '')
if acpp_lib_dir:
    config.substitutions.append(
        ('%acpp_link',
         '-L{0} -lacpp-rt -lacpp-common -Wl,-rpath={0} -pthread -ldl'.format(acpp_lib_dir)))
    config.available_features.add("pcuda")

# inhert env vars
config.environment = os.environ

config.available_features.add("always")
