#for doc only. do not execute directly !!!

exit

find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/\(tensorflow\|tf\)\.contrib\.layers/\1.keras.layers/g'

find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/\(tensorflow\|tf\)\.contrib /\1.keras /g'

find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/tf.train.AdamOptimizer/tf.optimizers.Adam/g'

find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/tf.variable_scope/tf.compat.v1.variable_scope/g'

find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/tf.placeholder/tf.compat.v1.placeholder/g'
find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/tf.get_variable/tf.compat.v1.get_variable/g'
find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/tf.layers/tf.compat.v1.layers/g'
find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/stable_baselines.common.tf.compat.v1.layers/stable_baselines.common.tf.layers/g'

find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/stable_baselines.common.tf\.layers/stable_baselines.common.tf_layers/g'


find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/ as tf\.compat\.v1\.layers/ as tf_layers/g'

find stable_baselines/ -name \*.py | xargs sed -i'' -e 's/tf.compat.v1.layers.fully_connected/tf.keras.layers.fully_connected/g'


