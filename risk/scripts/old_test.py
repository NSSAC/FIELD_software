import parsl
from parsl import python_app, join_app, Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from pdb import set_trace

# Initialize Parsl configuration
config = Config(
    executors=[
        HighThroughputExecutor(
            label="htex_local",
            max_workers_per_node=4,
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
)
parsl.load(config)

# Define your helper function at the top level
@python_app
def helper_function(x):
    return x * 2

# Define a Parsl app that uses the helper function
@join_app
def main_function(value):
    result = helper_function(value)
    return result

# Call the Parsl function
future = main_function(5)
print(future.result())

