def default_func(func):
    """Decorates a method to detect overrides in subclasses."""
    func._is_default = True
    return func
