"""
Tools module for the chat system
"""

import logging
import sys
from semantic_kernel.functions import kernel_function


logger = logging.getLogger(__name__)

class SumNumbersPlugin:
    @kernel_function
    def sum_numbers(self, a: float, b: float) -> float:
        """Adds two numbers together and returns the result."""
        result = a + b
        return result

class MultiplyNumbersPlugin:
    @kernel_function
    def multiply_numbers(self, a: float, b: float) -> float:
        """Multiplies two numbers together and returns the result."""
        result = a * b
        return result

def get_all_plugins():
    """Dynamically discover and return all plugin classes in this module."""
    import inspect
    
    plugins = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if (inspect.isclass(obj) and 
            name.endswith('Plugin') and 
            obj.__module__ == __name__):
            plugins.append(obj)
    
    return plugins


def get_all_plugin_instances():
    """Return instances of all available plugins."""
    return [plugin_class() for plugin_class in get_all_plugins()]
